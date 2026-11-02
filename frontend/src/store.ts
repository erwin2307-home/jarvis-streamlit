import { create } from "zustand";

export type JarvisState = "idle" | "listening" | "transcribe" | "thinking" | "speaking" | "error";

/** Discriminated union of all known inbound event types from the backend. */
export type InboundEvent =
  | { type: "state"; value: JarvisState }
  | { type: "transcript_partial"; text: string }
  | { type: "transcript_final"; text: string }
  | { type: "response_chunk"; text: string }
  | { type: "response_done"; full: string }
  | { type: "audio_level"; rms: number; fft: number[] }
  | { type: "error"; stage: string; msg: string; retriable: boolean };

interface StoreState {
  state: JarvisState;
  lastTranscript: string;
  lastResponse: string;
  audioLevel: number;
  connected: boolean;
  lastError: string;
  setConnected: (b: boolean) => void;
  handleEvent: (ev: InboundEvent) => void;
}

export const useJarvisStore = create<StoreState>((set) => ({
  state: "idle",
  lastTranscript: "",
  lastResponse: "",
  audioLevel: -120,
  connected: false,
  lastError: "",
  setConnected: (b) => set({ connected: b }),
  handleEvent: (ev) => set((s) => {
    switch (ev.type) {
      case "state":
        return { state: ev.value };
      case "transcript_partial":
        return { lastTranscript: ev.text };
      case "transcript_final":
        return { lastTranscript: ev.text };
      case "response_chunk":
        return { lastResponse: s.lastResponse + ev.text };
      case "response_done":
        return { lastResponse: ev.full };
      case "audio_level":
        return { audioLevel: ev.rms };
      case "error":
        return { state: "error", lastError: `${ev.stage}: ${ev.msg}` };
      default:
        return {};
    }
  }),
}));
