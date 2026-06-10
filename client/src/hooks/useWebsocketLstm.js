import { useEffect, useRef, useState, useCallback } from "react";

export default function useWebSocket(url) {
  const ws = useRef(null);
  const [lastMessage, setLastMessage] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  const connect = useCallback(() => {
    if (ws.current) return;

    ws.current = new WebSocket(url);

    ws.current.onopen = () => {
      setIsConnected(true);
    };

    ws.current.onclose = () => {
      setIsConnected(false);
      ws.current = null;
    };

    ws.current.onmessage = (event) => {
      setLastMessage(event.data);
    };
  }, [url]);

  const disconnect = useCallback(() => {
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }
    setIsConnected(false);
  }, []);

  const send = useCallback((data) => {
    if (ws.current?.readyState === 1) {
      ws.current.send(data);
    }
  }, []);

  return {
    connect,
    disconnect,
    send,
    lastMessage,
    isConnected,
  };
}
