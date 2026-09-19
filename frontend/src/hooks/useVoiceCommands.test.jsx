import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import useVoiceCommands from './useVoiceCommands'

// Minimal controllable WebSocket: tests drive open/message/close by hand.
class FakeSocket {
  static instances = []
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  constructor(url) {
    this.url = url
    this.readyState = FakeSocket.CONNECTING
    this.sent = []
    this.close = vi.fn(() => {
      this.readyState = FakeSocket.CLOSED
    })
    FakeSocket.instances.push(this)
  }
  send(data) {
    this.sent.push(data)
  }
  async open() {
    this.readyState = FakeSocket.OPEN
    await this.onopen?.()
  }
  message(payload) {
    this.onmessage?.({ data: JSON.stringify(payload) })
  }
  serverClose() {
    this.readyState = FakeSocket.CLOSED
    this.onclose?.()
  }
}

function fakeAudioContext() {
  return class {
    constructor() {
      this.sampleRate = 16000
      this.state = 'running'
      this.destination = {}
    }
    createMediaStreamSource() {
      return { connect: vi.fn() }
    }
    createScriptProcessor() {
      return { connect: vi.fn(), disconnect: vi.fn() }
    }
    createGain() {
      return { gain: {}, connect: vi.fn() }
    }
    close() {
      this.state = 'closed'
      return Promise.resolve()
    }
  }
}

const track = { stop: vi.fn() }
const getUserMedia = vi.fn()

beforeEach(() => {
  FakeSocket.instances = []
  track.stop.mockClear()
  getUserMedia.mockReset().mockResolvedValue({ getTracks: () => [track] })
  vi.stubGlobal('WebSocket', FakeSocket)
  vi.stubGlobal('MediaRecorder', class {})
  vi.stubGlobal('AudioContext', fakeAudioContext())
  Object.defineProperty(navigator, 'mediaDevices', { value: { getUserMedia }, configurable: true })
  window.requestAnimationFrame = vi.fn(() => 1)
  window.cancelAnimationFrame = vi.fn()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

async function startListening(props) {
  const hook = renderHook(() => useVoiceCommands(props))
  await act(async () => {
    await hook.result.current.start()
  })
  const socket = FakeSocket.instances[0]
  await act(async () => {
    await socket.open()
  })
  return { ...hook, socket }
}

describe('useVoiceCommands', () => {
  it('starts idle', () => {
    const { result } = renderHook(() => useVoiceCommands())
    expect(result.current.status).toBe('idle')
    expect(result.current.isListening).toBe(false)
  })

  it('reports an error when the browser has no microphone support', async () => {
    Object.defineProperty(navigator, 'mediaDevices', { value: undefined, configurable: true })
    const { result } = renderHook(() => useVoiceCommands())
    await act(async () => {
      await result.current.start()
    })
    expect(result.current.error).toMatch(/not supported/i)
    expect(FakeSocket.instances).toHaveLength(0)
  })

  it('connects with the stored token, url-encoded, over ws', async () => {
    localStorage.setItem('sinhaspeech_token', 'a b+c')
    const { result } = renderHook(() => useVoiceCommands())
    await act(async () => {
      await result.current.start()
    })
    const url = FakeSocket.instances[0].url
    expect(url).toMatch(/^ws/)
    expect(url).toContain('/streaming/ws?token=a%20b%2Bc')
    expect(result.current.status).toBe('connecting')
  })

  it('sends a COMMAND-mode start message and becomes listening once the mic opens', async () => {
    const { result, socket } = await startListening()
    expect(JSON.parse(socket.sent[0])).toEqual({ type: 'start', mode: 'COMMAND' })
    expect(result.current.status).toBe('listening')
    expect(result.current.isListening).toBe(true)
  })

  it('forwards recognised commands and command_maybe messages to the latest callbacks', async () => {
    const onCommand = vi.fn()
    const onCommandMaybe = vi.fn()
    const { socket } = await startListening({ onCommand, onCommandMaybe })
    act(() => {
      socket.message({ type: 'command', command: 'next' })
      socket.message({ type: 'command_maybe', fuzzy_command: 'save', embedding_command: null })
    })
    expect(onCommand).toHaveBeenCalledWith('next')
    expect(onCommandMaybe).toHaveBeenCalledWith(
      expect.objectContaining({ fuzzy_command: 'save', embedding_command: null }),
    )
  })

  it.each([
    ['NotAllowedError', /permission was denied/i],
    ['SecurityError', /permission was denied/i],
    ['NotFoundError', /no microphone/i],
    ['SomethingElse', /could not be started/i],
  ])('maps a %s from getUserMedia to a readable error', async (name, message) => {
    getUserMedia.mockRejectedValue(Object.assign(new Error('x'), { name }))
    const { result, socket } = await startListening()
    expect(result.current.status).toBe('error')
    expect(result.current.error).toMatch(message)
    expect(socket.close).toHaveBeenCalled()
  })

  it('surfaces a server error message', async () => {
    const { result, socket } = await startListening()
    act(() => socket.message({ type: 'error', message: 'Too many sessions' }))
    expect(result.current.error).toBe('Too many sessions')
  })

  it('stop() asks the server to end the session, and session_end returns to idle and releases the mic', async () => {
    const { result, socket } = await startListening()
    act(() => result.current.stop())
    expect(result.current.status).toBe('stopping')
    expect(JSON.parse(socket.sent.at(-1))).toEqual({ type: 'stop' })
    act(() => socket.message({ type: 'session_end', transcript_id: null }))
    expect(result.current.status).toBe('idle')
    expect(track.stop).toHaveBeenCalled()
  })

  it('an unexpected socket close is an error, but a close after stop() is not', async () => {
    const first = await startListening()
    act(() => first.socket.serverClose())
    expect(first.result.current.status).toBe('error')
    expect(first.result.current.error).toMatch(/connection was lost/i)

    FakeSocket.instances = []
    const second = await startListening()
    act(() => second.result.current.stop())
    act(() => second.socket.serverClose())
    expect(second.result.current.status).toBe('idle')
    expect(second.result.current.error).toBe('')
  })

  it('closes the socket and releases the mic when the component unmounts mid-listen', async () => {
    const { unmount, socket } = await startListening()
    unmount()
    expect(socket.close).toHaveBeenCalled()
    expect(track.stop).toHaveBeenCalled()
  })

  it('does not report a false error when unmounting aborts a still-connecting socket', async () => {
    const { result, unmount } = renderHook(() => useVoiceCommands())
    await act(async () => {
      await result.current.start()
    })
    const socket = FakeSocket.instances[0]
    unmount()
    // Aborting a CONNECTING socket fires onerror in real browsers.
    socket.onerror?.()
    socket.onclose?.()
    expect(socket.close).toHaveBeenCalled()
    expect(result.current.error).toBe('')
  })
})
