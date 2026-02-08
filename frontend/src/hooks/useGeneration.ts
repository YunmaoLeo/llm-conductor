import { useState, useCallback, useRef } from 'react';
import type { WsMessage, CompositionStatus, MusicFeatures } from '../types';

interface GenerationState {
  status: CompositionStatus;
  messages: WsMessage[];
  result: {
    generationId: string;
    midiUrl: string;
    textDescription: string;
    features: MusicFeatures | null;
    iterations: number;
  } | null;
  error: string | null;
}

export function useGeneration() {
  const [state, setState] = useState<GenerationState>({
    status: 'idle',
    messages: [],
    result: null,
    error: null,
  });
  const wsRef = useRef<WebSocket | null>(null);

  const compose = useCallback((intent: string) => {
    // Reset state
    setState({
      status: 'connecting',
      messages: [],
      result: null,
      error: null,
    });

    // Determine WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/compose`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setState((s) => ({ ...s, status: 'generating' }));
      ws.send(JSON.stringify({ intent }));
    };

    ws.onmessage = (event) => {
      const data: WsMessage = JSON.parse(event.data);

      setState((s) => {
        const newMessages = [...s.messages, data];
        let newStatus = s.status;
        let newResult = s.result;
        let newError = s.error;

        switch (data.type) {
          case 'evaluation':
            newStatus = 'evaluating';
            break;
          case 'tokens':
          case 'status':
            newStatus = 'generating';
            break;
          case 'complete':
            newStatus = 'complete';
            if (data.success && data.generation_id && data.midi_url) {
              newResult = {
                generationId: data.generation_id,
                midiUrl: data.midi_url,
                textDescription: data.text_description || '',
                features: data.features || null,
                iterations: data.iterations || 0,
              };
            }
            break;
          case 'error':
            if (!data.message?.includes('iteration')) {
              newError = data.message || 'Unknown error';
            }
            break;
        }

        return {
          status: newStatus,
          messages: newMessages,
          result: newResult,
          error: newError,
        };
      });
    };

    ws.onerror = () => {
      setState((s) => ({
        ...s,
        status: 'error',
        error: 'WebSocket connection failed',
      }));
    };

    ws.onclose = () => {
      wsRef.current = null;
    };
  }, []);

  const cancel = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setState((s) => ({ ...s, status: 'idle' }));
  }, []);

  return {
    ...state,
    compose,
    cancel,
    isGenerating: state.status === 'generating' || state.status === 'evaluating' || state.status === 'connecting',
  };
}
