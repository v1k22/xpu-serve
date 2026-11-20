# 0024: Realtime transcription (`/v1/realtime`)

Status: **accepted** · 2026-11-20

## Context

Speech recognition (0023) took a whole file per request. A live microphone needs the text of each thing said shortly
after it is said, while audio keeps arriving. vLLM and OpenAI both serve this as a WebSocket of JSON events; OpenAI's
Realtime API with a session of type "transcription" (GA) is the one clients are written against: audio appended as
base64 PCM, turns ended by the server's voice activity detection or by the client, each turn's transcript streamed
back as deltas.

## Decision

- **OpenAI's GA transcription session, as it is**: `GET /v1/realtime` (any `intent` but transcription refused), the
  `realtime` subprotocol when offered, the GA event names and shapes (`session.update` with `audio.input.format`,
  `.transcription`, `.turn_detection`; `input_audio_buffer.append / commit / clear`; `speech_started`,
  `speech_stopped`, `committed`, `conversation.item.added`, `...input_audio_transcription.delta / completed /
  failed`, `conversation.item.done`; `error` with the client's `event_id`). What is not supported is refused by name
  as an `error` event, and the session goes on: `semantic_vad`, `idle_timeout_ms`, noise reduction, logprobs,
  `keywords`, formats other than PCM at 24 kHz, response and conversation events.
- **A turn is transcribed whole**, through the same `InferenceService::transcribe` as a file: the same prompt, the
  same model code, the same result, so realtime needs nothing new from a model. Qwen3-ASR's encoder normalizes the
  log-mel features by the loudest frame of the clip, so windows encoded while a turn is still growing would not be
  exact; transcribing at the commit is, and a turn of a few seconds is encoded and prefilled in tens of
  milliseconds. Partial text before the end of a turn (OpenAI's events have none for transcription sessions) is left
  out.
- **The session is transport-free** (`server/realtime_session`): client events in, server events out through a
  callback, unit-tested without a socket; the WebSocket loop in `server.cpp` is a few lines. Committed turns queue to
  a thread of the session's own, so the connection keeps reading audio while one is transcribed; closing the
  connection stops the transcription in progress.
- **The server VAD is signal code, not a model** (`audio/vad`): 20 ms frames, speech when a frame's level clears an
  absolute level (`threshold` 0 to 1 maps to -55 to -20 dBFS) and a tracked noise floor by 10 dB; 60 ms of speech
  starts a turn, `silence_duration_ms` of silence ends it, `prefix_padding_ms` of audio is kept before it. The first
  200 ms of a stream only learn the noise floor, so a loud room is not one long turn. A model-based VAD (Silero's)
  would be a second model to load; the energy detector is enough where speech is louder than the room, and the
  client can always commit turns itself.
- **`TranscriptionRequest` holds a decoded clip**: the HTTP handler decodes uploads (a refusal is a 400 naming
  `file`), the realtime session builds a clip from its PCM at 24 kHz, and the model resamples as it does any clip.
- **The API key** may come as OpenAI's browser subprotocol `openai-insecure-api-key.<key>` beside the header, since a
  browser cannot set headers on a WebSocket.

## What was measured

Qwen3-ASR-1.7B q8 on an Arc Pro B60, LibriSpeech read aloud at real time in 100 ms pieces: after the VAD ends a turn
(500 ms into the silence), the first text about 50 ms and the whole transcript 130 to 250 ms later for turns of 3 to
7 s; with the client committing, 55 to 150 ms and 145 to 780 ms (5 to 29 s of speech). The same session, through a
proxy and a tunnel, from a browser page.
