import { useRef, useCallback } from "react";

export function useWebSocket(onMessage) {
  const wsRef = useRef(null);

  const connectWebSocket = useCallback(() => {
    //const ws = new WebSocket("ws://localhost:8000/ws");
    const ws = new WebSocket("wss://kcrk46sc-8000.asse.devtunnels.ms/ws");
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      onMessage(event.data);
    };

    ws.onerror = (err) => {
      console.log("WebSocket error", err);
    };

    ws.onclose = () => {
      console.log("WebSocket closed");
    };
  }, [onMessage]);

  const disconnectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  return {
    connectWebSocket,
    disconnectWebSocket,
    wsRef,
  };
}
