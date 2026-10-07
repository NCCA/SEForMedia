import pytest
import torch
from torch import nn
import torch.nn.functional as F

from ImageToImage import image_models as core


@pytest.fixture(params=["transpose", "resize"])
def unet(request: pytest.FixtureRequest) -> core.UNet:
    return core.UNet(out_channels=1, upsampling=request.param)


@pytest.fixture
def image_and_trimap() -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.rand(3, 145, 173)
    trimap = torch.full((1, 145, 173), 2, dtype=torch.uint8)
    trimap[:, 20:120, 30:140] = 1
    trimap[:, 19, :] = 3
    image[0] = (trimap[0] == 1).float()
    return image, trimap


@pytest.fixture
def segmentation_example() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    valid = torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]])
    logits = (target * 2 - 1) * 12
    return logits, target, valid


def test_unet_preserves_odd_spatial_dimensions(unet: core.UNet) -> None:
    prediction = unet(torch.rand(1, 3, 33, 41))

    assert prediction.shape == (1, 1, 33, 41)


def test_unet_has_about_two_million_parameters(unet: core.UNet) -> None:
    assert 1_500_000 < sum(p.numel() for p in unet.parameters()) < 2_500_000


def test_unet_backpropagates_to_encoder_and_uses_skips(unet: core.UNet) -> None:
    image = torch.rand(1, 3, 33, 41)
    unet(image).square().mean().backward()

    assert unet.encoders[0][0].weight.grad.abs().sum().item() > 0
    with torch.no_grad():
        assert not torch.allclose(unet(image), unet(image, use_skips=False))


@pytest.mark.parametrize("model_type", [core.ESPCN, core.SuperResolutionUNet])
def test_super_resolution_doubles_spatial_dimensions(
    model_type: type[nn.Module],
) -> None:
    model = model_type()

    assert model(torch.rand(1, 3, 17, 21)).shape == (1, 3, 34, 42)


def test_espcn_has_fewer_than_thirty_thousand_parameters() -> None:
    assert sum(p.numel() for p in core.ESPCN().parameters()) < 30_000


def test_pixel_shuffle_moves_channels_to_spatial_positions() -> None:
    values = torch.arange(4.0).reshape(1, 4, 1, 1)

    torch.testing.assert_close(
        nn.PixelShuffle(2)(values)[0, 0], torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    )


def test_subtracting_noise_target_restores_clean_patch(
    image_and_trimap: tuple[torch.Tensor, torch.Tensor],
) -> None:
    image, trimap = image_and_trimap
    noisy, noise, _ = core.make_patch(
        image, trimap, "denoise", torch.Generator().manual_seed(8)
    )
    clean, _, _ = core.make_patch(
        image, trimap, "segment", torch.Generator().manual_seed(8)
    )

    assert noise.shape[-2:] == (128, 128)
    torch.testing.assert_close(noisy - noise, clean)


def test_super_resolution_patch_is_downsampled_from_its_target(
    image_and_trimap: tuple[torch.Tensor, torch.Tensor],
) -> None:
    image, trimap = image_and_trimap
    low, target, _ = core.make_patch(
        image, trimap, "sr", torch.Generator().manual_seed(8)
    )

    assert low.shape == (3, 64, 64)
    assert target.shape[-2:] == (128, 128)
    torch.testing.assert_close(
        low, F.interpolate(target[None], scale_factor=0.5, mode="area")[0]
    )


def test_segmentation_patch_keeps_image_and_binary_mask_aligned(
    image_and_trimap: tuple[torch.Tensor, torch.Tensor],
) -> None:
    image, trimap = image_and_trimap
    inputs, target, valid = core.make_patch(
        image, trimap, "segment", torch.Generator().manual_seed(8)
    )

    assert target.shape[-2:] == (128, 128)
    torch.testing.assert_close(inputs[:1], target)
    assert ((target == 0) | (target == 1)).all()
    assert (valid == 0).any()


def test_segmentation_loss_and_scores_ignore_invalid_pixels(
    segmentation_example: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    logits, target, valid = segmentation_example
    initial_loss = core.segmentation_loss(logits, target, valid)
    logits[0, 0, 1, 0] = 100

    torch.testing.assert_close(
        initial_loss, core.segmentation_loss(logits, target, valid)
    )
    assert initial_loss.item() < 0.001
    assert core.mask_scores(logits.sigmoid(), target, valid) == (1.0, 1.0)


def test_all_ignored_segmentation_patch_has_zero_loss(
    segmentation_example: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    logits, target, valid = segmentation_example

    assert core.segmentation_loss(logits, target, torch.zeros_like(valid)).item() == 0.0


def test_identical_images_have_zero_mse_and_infinite_psnr() -> None:
    image = torch.zeros(3, 12, 14)

    assert core.image_scores(image, image) == (0.0, float("inf"))


def test_image_scores_match_known_uniform_error() -> None:
    image = torch.zeros(3, 12, 14)
    mse, psnr = core.image_scores(image, image + 0.1)

    assert mse == pytest.approx(0.01, abs=1e-6)
    assert psnr == pytest.approx(20.0, abs=1e-5)


@pytest.mark.parametrize("shape", [(3, 7, 9), (3, 129, 173), (3, 128, 128)])
@pytest.mark.parametrize("scale", [1, 2])
def test_tiles_cover_edges_at_each_output_scale(
    shape: tuple[int, int, int], scale: int
) -> None:
    image = torch.rand(shape)
    model = nn.Upsample(scale_factor=scale, mode="nearest")
    result = core.tiled_predict(model, image, tile=64, overlap=16, scale=scale)
    expected = image.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1)

    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
    assert result.device.type == "cpu"


def test_tiling_rejects_overlap_equal_to_tile_size() -> None:
    with pytest.raises(ValueError, match="0 <= overlap < tile"):
        core.tiled_predict(nn.Identity(), torch.rand(3, 128, 128), tile=32, overlap=32)
