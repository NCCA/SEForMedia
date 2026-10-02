/// <reference path="./marimo-studio.d.ts" />

import { Deck, Slide, Stack } from "@revealjs/react";
import { useEffect, useRef } from "react";
import type { RevealApi } from "reveal.js";
import "reveal.js/reveal.css";

import { useMarimoValue } from "./lib/use-marimo-value.ts";

const VIEW_HEADING = "";
const NOTEBOOK_TITLE = "Nnfromscratchslides";

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

/** The notebook's `deck` value: one list of cell names per topic. */
type Topics = string[][];

const isTopics = (value: unknown): value is Topics =>
  Array.isArray(value) &&
  value.every(
    (topic) =>
      Array.isArray(topic) && topic.every((name) => typeof name === "string"),
  );

/** One notebook cell on its own slide. The name comes from the notebook at
 * runtime, so the host opts in to targets Studio cannot see in the source. */
const CellSlide = ({ name }: { name: string }) => (
  <Slide>
    <marimo-cell name={name} data-marimo-allow="*" />
  </Slide>
);

export const App = () => {
  const deck = useSettledLayout();
  const { hostRef, value } = useMarimoValue<Topics>("deck");
  const topics = isTopics(value) ? value.filter((topic) => topic.length > 0) : [];
  return (
    <>
      <span ref={hostRef} hidden mo-value="deck" />
      <Deck
        className="studio-deck"
        deckRef={deck}
        config={{
          controls: true,
          controlsLayout: "bottom-right",
          keyboardCondition,
          progress: true,
          scrollActivationWidth: 0,
          transition: "slide",
        }}
      >
        <Slide>
          <p className="deck-kicker">{VIEW_HEADING}</p>
          <h1>Neural Network from Scratch</h1>
        </Slide>
        {topics.map((topic) =>
          topic.length === 1
            ? <CellSlide key={topic[0]} name={topic[0]} />
            : (
              <Stack key={topic[0]}>
                {topic.map((name) => <CellSlide key={name} name={name} />)}
              </Stack>
            )
        )}
      </Deck>
    </>
  );
};
