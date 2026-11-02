import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { useJarvisStore } from "../store";
import { HudShell } from "../hud/HudShell";

describe("HudShell", () => {
  beforeEach(() => {
    useJarvisStore.setState({
      state: "idle",
      lastTranscript: "",
      lastResponse: "",
      audioLevel: -120,
      connected: false,
    });
  });

  it("renders state in StatusCorners", () => {
    useJarvisStore.setState({ state: "listening", lastTranscript: "", lastResponse: "", audioLevel: -42, connected: true });
    render(<HudShell />);
    // Assert specifically on the state label in the top-left corner
    expect(screen.getByText(/JARVIS · v0\.1 · LISTENING/i)).toBeInTheDocument();
  });

  it("renders last response when speaking", () => {
    useJarvisStore.setState({ state: "speaking", lastTranscript: "", lastResponse: "Hallo Erwin.", audioLevel: -30, connected: true });
    render(<HudShell />);
    expect(screen.getByText("Hallo Erwin.")).toBeInTheDocument();
  });

  it("shows a bottom-center status confirming the question was understood while thinking", () => {
    useJarvisStore.setState({
      state: "thinking",
      lastTranscript: "wie spät ist es",
      lastResponse: "",
      audioLevel: -40,
      connected: true,
    });
    render(<HudShell />);
    expect(screen.getByText(/verstanden.*wie spät ist es.*generiert/i)).toBeInTheDocument();
  });

  it("does not show the thinking status banner outside the thinking state", () => {
    useJarvisStore.setState({
      state: "speaking",
      lastTranscript: "wie spät ist es",
      lastResponse: "14 Uhr.",
      audioLevel: -40,
      connected: true,
    });
    render(<HudShell />);
    expect(screen.queryByText(/verstanden/i)).not.toBeInTheDocument();
  });
});
