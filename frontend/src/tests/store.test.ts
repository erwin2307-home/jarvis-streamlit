import { describe, it, expect, beforeEach } from "vitest";
import { useJarvisStore } from "../store";

describe("jarvis store", () => {
  beforeEach(() => {
    useJarvisStore.setState({
      state: "idle",
      lastTranscript: "",
      lastResponse: "",
      audioLevel: -120,
      connected: false,
    });
  });

  it("updates state on state event", () => {
    useJarvisStore.getState().handleEvent({ type: "state", value: "listening" });
    expect(useJarvisStore.getState().state).toBe("listening");
  });

  it("updates transcript on transcript_final", () => {
    useJarvisStore.getState().handleEvent({ type: "transcript_final", text: "hallo Jarvis" });
    expect(useJarvisStore.getState().lastTranscript).toBe("hallo Jarvis");
  });

  it("appends response chunks then finalizes", () => {
    useJarvisStore.getState().handleEvent({ type: "response_chunk", text: "Hallo " });
    useJarvisStore.getState().handleEvent({ type: "response_chunk", text: "Erwin" });
    expect(useJarvisStore.getState().lastResponse).toBe("Hallo Erwin");
    useJarvisStore.getState().handleEvent({ type: "response_done", full: "Hallo Erwin." });
    expect(useJarvisStore.getState().lastResponse).toBe("Hallo Erwin.");
  });

  it("updates audio level", () => {
    useJarvisStore.getState().handleEvent({ type: "audio_level", rms: -42, fft: [] });
    expect(useJarvisStore.getState().audioLevel).toBe(-42);
  });

  it("ignores unknown event types", () => {
    const before = useJarvisStore.getState();
    // @ts-expect-error testing unknown type
    useJarvisStore.getState().handleEvent({ type: "nonsense" });
    expect(useJarvisStore.getState()).toEqual(before);
  });
});
