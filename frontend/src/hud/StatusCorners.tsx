import React from "react";
import { useJarvisStore } from "../store";

const cornerStyle: React.CSSProperties = {
  position: "absolute", color: "#00ddff", font: "11px monospace",
  padding: 12, opacity: 0.85, letterSpacing: 1.2,
};

export function StatusCorners() {
  const state = useJarvisStore((s) => s.state);
  const connected = useJarvisStore((s) => s.connected);
  const lastTranscript = useJarvisStore((s) => s.lastTranscript);
  const lastResponse = useJarvisStore((s) => s.lastResponse);
  const label = state.toUpperCase();

  return (
    <>
      <div style={{ ...cornerStyle, top: 0, left: 0 }}>
        JARVIS · v0.1 · {label}
      </div>
      <div style={{ ...cornerStyle, top: 0, right: 0 }}>
        {connected ? "ONLINE" : "DISCONNECTED"}
      </div>
      <div style={{ ...cornerStyle, bottom: 0, left: 0, maxWidth: "40%" }}>
        {lastTranscript ? `YOU · ${lastTranscript}` : "LISTENING ·····"}
      </div>
      <div style={{ ...cornerStyle, bottom: 0, right: 0, maxWidth: "40%", textAlign: "right" }}>
        {lastResponse || "de-DE · thorsten"}
      </div>
      {state === "thinking" && (
        <div
          style={{
            position: "absolute", bottom: 32, left: "50%", transform: "translateX(-50%)",
            color: "#00ddff", font: "12px monospace", letterSpacing: 1.2,
            opacity: 0.9, textAlign: "center", maxWidth: "70%",
          }}
        >
          ✓ Verstanden: „{lastTranscript}" — Antwort wird generiert…
        </div>
      )}
    </>
  );
}
