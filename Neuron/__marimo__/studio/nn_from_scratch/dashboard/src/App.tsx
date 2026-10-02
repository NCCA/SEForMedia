/// <reference path="./marimo-studio.d.ts" />

import { Deck, Slide } from "@revealjs/react";
import { useEffect, useRef } from "react";
import type { RevealApi } from "reveal.js";
import "reveal.js/reveal.css";

const VIEW_HEADING = "Dashboard";
const NOTEBOOK_TITLE = "Neural Networks From Scratch";

const keyboardCondition = (event: KeyboardEvent) =>
  !event.composedPath().some(
    (target) =>
      target instanceof Element && target.matches("marimo-cell, marimo-output"),
  );

/** Center slides again once projected notebook content has its final size. */
const useSettledLayout = () => {
  const deck = useRef<RevealApi | null>(null);
  useEffect(() => {
    const layout = () => deck.current?.layout();
    document.addEventListener("marimo-studio:idle", layout);
    return () => document.removeEventListener("marimo-studio:idle", layout);
  }, []);
  return deck;
};

export const App = () => {
  const deck = useSettledLayout();
  return (
    <Deck
      className="studio-deck"
      deckRef={deck}
      config={{
        controls: true,
        keyboardCondition,
        progress: true,
        scrollActivationWidth: 0,
        transition: "slide",
      }}
    >
      <Slide>
        <p className="deck-kicker">{VIEW_HEADING}</p>
        <h1>{NOTEBOOK_TITLE}</h1>
      </Slide>
      {[
        {
          "target": "cell-3",
          "title": "Neural Networks From Scratch",
          "showTitle": false,
        },
        {
          "target": "cell-4",
          "title": "Stage 1 — A Single Neuron (fixed weights)",
          "showTitle": false,
        },
        {
          "target": "cell-6",
          "title": "Cell 6",
          "showTitle": true,
        },
        {
          "target": "cell-9",
          "title": "Cell 9",
          "showTitle": true,
        },
        {
          "target": "cell-11",
          "title": "Cell 11",
          "showTitle": true,
        },
        {
          "target": "cell-13",
          "title": "Stage 2 — Learning the weights (gradient descent)",
          "showTitle": false,
        },
        {
          "target": "cell-15",
          "title": "TODO: implement compute_gradient",
          "showTitle": false,
        },
        {
          "target": "cell-17",
          "title": "Cell 17",
          "showTitle": true,
        },
        {
          "target": "cell-18",
          "title": "Cell 18",
          "showTitle": true,
        },
        {
          "target": "cell-20",
          "title": "Cell 20",
          "showTitle": true,
        },
        {
          "target": "cell-21",
          "title": "Cell 21",
          "showTitle": true,
        },
        {
          "target": "cell-22",
          "title": "Stage 3 — Where a single neuron breaks",
          "showTitle": false,
        },
        {
          "target": "cell-23",
          "title": "Stage 4 — A two-layer network solves XOR",
          "showTitle": false,
        },
        {
          "target": "cell-25",
          "title": "TODO: implement computegradientsmlp",
          "showTitle": false,
        },
        {
          "target": "cell-27",
          "title": "Cell 27",
          "showTitle": true,
        },
        {
          "target": "cell-28",
          "title": "Cell 28",
          "showTitle": true,
        },
        {
          "target": "cell-31",
          "title": "Cell 31",
          "showTitle": true,
        },
        {
          "target": "cell-32",
          "title": "Cell 32",
          "showTitle": true,
        },
        {
          "target": "cell-33",
          "title": "Wrap-up",
          "showTitle": false,
        },
      ].map(({ target, title, showTitle }) => (
        <Slide key={target}>
          {showTitle ? <h2>{title}</h2> : null}
          <marimo-cell name={target} />
        </Slide>
      ))}
    </Deck>
  );
};
