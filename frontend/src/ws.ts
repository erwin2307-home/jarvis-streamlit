import type { InboundEvent } from "./store";

export interface JarvisWsOptions {
  url: string;
  onEvent: (ev: InboundEvent) => void;
  onConnected?: () => void;
  onDisconnected?: () => void;
  reconnectMs?: number;
}

export class JarvisWs {
  private socket: WebSocket | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private closed = false;
  constructor(private opts: JarvisWsOptions) {}

  connect() {
    this.closed = false;
    this.socket = new WebSocket(this.opts.url);
    this.socket.onopen = () => this.opts.onConnected?.();
    this.socket.onmessage = (ev) => {
      try {
        this.opts.onEvent(JSON.parse(ev.data));
      } catch {
        // swallow malformed frames; backend should never emit them
      }
    };
    this.socket.onclose = () => {
      this.opts.onDisconnected?.();
      if (!this.closed) {
        this.timer = setTimeout(() => this.connect(), this.opts.reconnectMs ?? 1000);
      }
    };
    this.socket.onerror = () => this.socket?.close();
  }

  send(obj: object) {
    // readyState 1 === OPEN; avoid referencing WebSocket.OPEN so mocks work in tests
    if (this.socket && this.socket.readyState === 1) {
      this.socket.send(JSON.stringify(obj));
    }
  }

  close() {
    this.closed = true;
    if (this.timer) clearTimeout(this.timer);
    this.socket?.close();
  }
}
