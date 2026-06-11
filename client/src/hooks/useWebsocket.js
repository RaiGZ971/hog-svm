import { useRef, useCallback } from "react";

export function useWebSocket(setDetectedGesture) {
  const wsRef = useRef(null);

  const connectWebSocket = useCallback(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      setDetectedGesture(event.data);
    };

    ws.onerror = (err) => {
      console.log("WebSocket error", err);
    };

    ws.onclose = () => {
      console.log("WebSocket closed");
    };
  }, [setDetectedGesture]);

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

