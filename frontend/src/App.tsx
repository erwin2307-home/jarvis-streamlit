import { useEffect, useRef } from "react";
import { HudShell } from "./hud/HudShell";
import { useJarvisStore } from "./store";
import { JarvisWs } from "./ws";

export function App() {
  const wsRef = useRef<JarvisWs | null>(null);

  useEffect(() => {
    const ws = new JarvisWs({
      url: "ws://127.0.0.1:8765/ws",
      onEvent: (ev) => useJarvisStore.getState().handleEvent(ev),
      onConnected: () => useJarvisStore.getState().setConnected(true),
      onDisconnected: () => useJarvisStore.getState().setConnected(false),
    });
    wsRef.current = ws;
    ws.connect();

    let unlisten: (() => void) | null = null;
    (async () => {
      try {
        const tauriEvent = await import("@tauri-apps/api/event");
        unlisten = await tauriEvent.listen("hotkey", () => {
          ws.send({ type: "manual_start" });
        });
      } catch {
        // Plain browser (vitest, vite dev without Tauri) — silently skip.
      }
    })();

    const onKeyDown = (e: KeyboardEvent) => {
      // Space = push-to-talk. Local window listener, not an OS-level global
      // hotkey — avoids the Win+J-style "already registered" conflict, and
      // guarantees a manual_start reaches the backend even when wake-word
      // detection doesn't fire.
      if (e.key === " " && !e.repeat) {
        e.preventDefault();
        ws.send({ type: "manual_start" });
      }
    };
    window.addEventListener("keydown", onKeyDown);

    return () => {
      ws.close();
      unlisten?.();
      window.removeEventListener("keydown", onKeyDown);
    };
  }, []);

  const onClickCore = () => wsRef.current?.send({ type: "manual_start" });

  return (
    <div onClick={onClickCore} style={{ cursor: "pointer", height: "100vh" }}>
      <HudShell />
    </div>
  );
}
