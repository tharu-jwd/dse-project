import { useEffect, useMemo, useRef, useState } from 'react'
import { api, downloadBlob } from '../api'
import { useAccessibility } from '../contexts/AccessibilityContext'
import { useToast } from '../contexts/ToastContext'
import { useVoiceCommandHandler, useVoiceCommandHint } from '../contexts/VoiceCommandContext'
import Icon from './Icon'
import { Alert, ConfirmDialog, StatusBadge } from './UI'

const formatTime = (seconds) =>
  `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`

export function ConfidenceText({ segment, threshold }) {
  if (!segment.words?.length) {
    // A confidence of exactly 0 with no per-word data means no score was
    // ever recorded for this segment (e.g. live/note recordings) rather than
    // a genuinely bad transcription, so it should not be flagged as low.
    const flagged = segment.confidence > 0 && segment.confidence < threshold
    return (
      <span className={flagged ? 'low-confidence' : ''}>
        {segment.text}
        {flagged && <span className="sr-only"> (low confidence)</span>}
      </span>
    )
  }
  return (
    <>
      {segment.words.map((word, index) => (
        <span
          key={`${word.text}-${index}`}
          className={word.confidence < threshold ? 'low-confidence' : ''}
        >
          {word.confidence < threshold && (
            <span className="confidence-mark" aria-hidden="true">
              ?
            </span>
          )}
          {word.text}
          {index < segment.words.length - 1 ? ' ' : ''}
          {word.confidence < threshold && <span className="sr-only"> (low confidence)</span>}
        </span>
      ))}
    </>
  )
}

export default function TranscriptEditor({
  initialTranscript,
  onTranscriptChange,
  compact = false,
}) {
  const [transcript, setTranscript] = useState(initialTranscript)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [confirmFinalize, setConfirmFinalize] = useState(false)
  const { confidenceThreshold, updatePreference } = useAccessibility()
  const { showToast } = useToast()
  const player = useRef(null)
  const [playbackUrl, setPlaybackURL] = useState('')
  const [mediaError, setMediaError] = useState('')
  const [searchWord, setSearchWord] = useState('')
  const [replaceWord, setReplaceWord] = useState('')
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [lastFormat, setLastFormat] = useState('')
  const [commandFeedback, setCommandFeedback] = useState('')
  const feedbackTimeoutRef = useRef(null)
  const showCommandFeedback = (message) => {
    window.clearTimeout(feedbackTimeoutRef.current)
    setCommandFeedback(message)
    feedbackTimeoutRef.current = window.setTimeout(() => setCommandFeedback(''), 2200)
  }
  useEffect(() => () => window.clearTimeout(feedbackTimeoutRef.current), [])
  // Compact mode is the per-question answer review nested inside a quiz
  // page - that page already owns the shared voice session's command
  // handler (next/previous/submit), so this instance registers nothing
  // rather than fighting it for the same handler slot. See
  // VoiceCommandContext for why there's only ever one session app-wide.
  useVoiceCommandHandler(
    compact
      ? null
      : (command) => {
          if (transcript.status === 'FINALIZED') return
          if (command === 'save') {
            if (!dirty) {
              showCommandFeedback('Nothing to save')
              return
            }
            showCommandFeedback('Saving…')
            save()
          } else if (command === 'submit') {
            showCommandFeedback('Opening finalize confirmation…')
            setConfirmFinalize(true)
          }
        },
  )
  useVoiceCommandHint(compact ? null : 'Say "save" or "submit"')
  useEffect(() => setTranscript(initialTranscript), [initialTranscript])
  useEffect(() => {
    let active = true
    let objectUrl = ''

    setPlaybackURL('')
    setMediaError('')

    if (!transcript.mediaUrl) {
      return () => {
        active = false
      }
    }

    if (
      transcript.mediaUrl.startsWith('blob:') ||
      transcript.mediaUrl.startsWith('data:')
    ) {
      setPlaybackURL(transcript.mediaUrl)

      return () => {
        active = false
      }
    }

    api
      .getMedia(transcript.mediaUrl)
      .then((blob) => {
        if (!active) return

        objectUrl = URL.createObjectURL(blob)
        setPlaybackURL(objectUrl)
      })
      .catch((cause) => {
        if (active) {
          setMediaError(
            cause.message || 'The recording could not be loaded.'
          )
        }
      })

      return () => {
        active = false
        if (objectUrl) {
          URL.revokeObjectURL(objectUrl)
        }
      }
  }, [transcript.mediaUrl])
  useEffect(() => {
    setIsPlaying(false)
    setCurrentTime(0)
    setDuration(0)
  }, [playbackUrl])
  useEffect(() => {
    const media = player.current
    if (!media) return
    const onTime = () => setCurrentTime(media.currentTime)
    const onLoaded = () => setDuration(media.duration || 0)
    const onPlay = () => setIsPlaying(true)
    const onPause = () => setIsPlaying(false)
    media.addEventListener('timeupdate', onTime)
    media.addEventListener('loadedmetadata', onLoaded)
    media.addEventListener('play', onPlay)
    media.addEventListener('pause', onPause)
    media.addEventListener('ended', onPause)
    return () => {
      media.removeEventListener('timeupdate', onTime)
      media.removeEventListener('loadedmetadata', onLoaded)
      media.removeEventListener('play', onPlay)
      media.removeEventListener('pause', onPause)
      media.removeEventListener('ended', onPause)
    }
  }, [playbackUrl])

  const editSegment = (id, text) => {
    setTranscript((current) => ({
      ...current,
      segments: current.segments.map((segment) =>
        segment.id === id ? { ...segment, text } : segment,
      ),
    }))
    setDirty(true)
    setError('')
  }
  const save = async () => {
    setSaving(true)
    setError('')
    try {
      const saved = await api.updateTranscript(transcript.id, {
        title: transcript.title,
        segments: transcript.segments,
      })
      setTranscript(saved)
      setDirty(false)
      onTranscriptChange?.(saved)
      showToast('Transcript changes saved.')
    } catch (cause) {
      setError(cause.message || 'Changes could not be saved. Your edits are still here.')
    } finally {
      setSaving(false)
    }
  }
  const finalize = async () => {
    setConfirmFinalize(false)
    setSaving(true)
    setError('')
    try {
      if (dirty)
        await api.updateTranscript(transcript.id, {
          title: transcript.title,
          segments: transcript.segments,
        })
      const result = await api.finalizeTranscript(transcript.id)
      setTranscript(result)
      setDirty(false)
      onTranscriptChange?.(result)
      showToast('Transcript finalized successfully.')
    } catch (cause) {
      setError(cause.message)
    } finally {
      setSaving(false)
    }
  }
  const exportFile = async (format) => {
    try {
      const blob = await api.exportTranscript(transcript.id, format)
      downloadBlob(
        blob,
        `${transcript.title.replace(/[^a-zA-Z0-9\u0D80-\u0DFF]+/g, '-')}.${format}`,
      )
      setLastFormat(format)
      showToast(`${format.toUpperCase()} export downloaded.`)
    } catch (cause) {
      setError(cause.message)
    }
  }
  const seek = (seconds) => {
    if (player.current && playbackUrl) {
      player.current.currentTime = seconds
      player.current.play().catch(() => {})
    }
  }
  const togglePlay = () => {
    if (!player.current) return
    if (player.current.paused) player.current.play().catch(() => {})
    else player.current.pause()
  }
  const scrub = (seconds) => {
    if (player.current) player.current.currentTime = seconds
    setCurrentTime(seconds)
  }
  const exportFormats = transcript.type === 'LECTURE' ? ['txt', 'docx', 'pdf'] : ['txt', 'docx']
  const stats = useMemo(() => {
    const words = transcript.segments.reduce(
      (sum, segment) => sum + segment.text.trim().split(/\s+/).filter(Boolean).length,
      0,
    )
    const lastSegment = transcript.segments[transcript.segments.length - 1]
    const seconds = lastSegment ? lastSegment.startTime : 0
    return {
      words,
      duration: formatTime(seconds),
      readingMinutes: Math.max(1, Math.round(words / 200)),
    }
  }, [transcript.segments])
  const executeReplace = () => {
    if (!searchWord.trim()) return
    const pattern = new RegExp(searchWord.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi')
    setTranscript((current) => ({
      ...current,
      segments: current.segments.map((segment) => ({
        ...segment,
        text: segment.text.replace(pattern, replaceWord),
      })),
    }))
    setDirty(true)
  }

  return (
    <div className={`transcript-editor ${compact ? 'transcript-editor--compact' : ''}`}>
      {!compact && (
        <div className="editor-toolbar">
          <div>
            <label htmlFor="transcript-title" className="sr-only">
              Transcript title
            </label>
            <input
              id="transcript-title"
              className="title-input"
              value={transcript.title}
              onChange={(e) => {
                setTranscript((current) => ({ ...current, title: e.target.value }))
                setDirty(true)
              }}
            />
            <div className="editor-meta">
              <StatusBadge status={transcript.status} />
              <span>{transcript.segments.length} segments</span>
              {dirty && <span className="unsaved-dot">● Unsaved changes</span>}
            </div>
          </div>
          <div className="editor-toolbar__actions">
            {!compact && (
              <div className="export-toggle" role="group" aria-label="Export format">
                {exportFormats.map((format) => (
                  <button
                    type="button"
                    key={format}
                    className={format === lastFormat ? 'active' : ''}
                    onClick={() => exportFile(format)}
                    disabled={dirty}
                    title={dirty ? 'Save changes before exporting' : `Export ${format.toUpperCase()}`}
                  >
                    {format.toUpperCase()}
                  </button>
                ))}
              </div>
            )}
            <button
              type="button"
              className="button button--secondary"
              disabled={saving || !dirty || transcript.status === 'FINALIZED'}
              onClick={save}
            >
              {saving ? (
                <span className="spinner spinner--small" />
              ) : (
                <Icon name="check" size={17} />
              )}{' '}
              Save draft
            </button>
            {transcript.status !== 'FINALIZED' && (
              <button
                type="button"
                className="button button--primary"
                disabled={saving}
                onClick={() => setConfirmFinalize(true)}
              >
                Finalize
              </button>
            )}
          </div>
        </div>
      )}
      {error && (
        <Alert>
          {error} {dirty && 'Your unsaved edits have been preserved.'}
        </Alert>
      )}
      {commandFeedback && (
        <p className="command-feedback command-feedback--success" role="status">
          <Icon name="check" size={14} /> {commandFeedback}
        </p>
      )}
      {!compact && (
        <div className={`player-bar ${!playbackUrl ? 'player-bar--empty' : ''}`}>
          {playbackUrl ? (
            <>
              {transcript.mediaType?.startsWith('video/') ? (
                <video
                  ref={player}
                  src={playbackUrl}
                  className="player-bar__video"
                  onClick={togglePlay}
                />
              ) : (
                <audio ref={player} src={playbackUrl} style={{ display: 'none' }} />
              )}
              <button
                type="button"
                className="player-bar__play"
                onClick={togglePlay}
                aria-label={isPlaying ? 'Pause' : 'Play'}
              >
                <Icon name={isPlaying ? 'stop' : 'play'} size={18} />
              </button>
              <div className="player-bar__track">
                <div
                  className="player-bar__fill"
                  style={{ width: duration ? `${(currentTime / duration) * 100}%` : '0%' }}
                />
                <input
                  type="range"
                  min={0}
                  max={duration || 0}
                  step={0.1}
                  value={currentTime}
                  onChange={(e) => scrub(Number(e.target.value))}
                  aria-label="Seek recording"
                />
              </div>
              <span className="player-bar__time">
                {formatTime(currentTime)} / {formatTime(duration)}
              </span>
            </>
          ) : (
            <>
              <Icon name={mediaError ? 'alert' : 'play'} size={18} />
              <span className="player-bar__message">
                {mediaError ||
                  (transcript.mediaUrl ? 'Loading recording…' : 'No recording is attached')}
              </span>
            </>
          )}
        </div>
      )}
      <div className="editor-layout">
        <section className="segments" aria-label="Transcript segments">
          {transcript.segments.map((segment, index) => {
            const isLow = segment.confidence > 0 && segment.confidence < confidenceThreshold
            return (
              <article className={`segment ${isLow ? 'segment--review' : ''}`} key={segment.id}>
                <div className="segment__gutter">
                  <button
                    type="button"
                    className="timestamp"
                    onClick={() => seek(segment.startTime)}
                    aria-label={`Play from ${formatTime(segment.startTime)}`}
                    title={`Play from ${formatTime(segment.startTime)}`}
                  >
                    {formatTime(segment.startTime)}
                  </button>
                  {isLow && (
                    <span
                      className="confidence-flag"
                      title={`${Math.round(segment.confidence * 100)}% confidence — needs review`}
                    >
                      <Icon name="alert" size={13} />
                      <span className="sr-only">
                        {Math.round(segment.confidence * 100)}% confidence, needs review
                      </span>
                    </span>
                  )}
                  <span className="sr-only">Segment {index + 1}</span>
                </div>
                <div className="segment__body">
                  <div className="confidence-preview" lang="si">
                    <ConfidenceText segment={segment} threshold={confidenceThreshold} />
                  </div>
                  <label htmlFor={`segment-${segment.id}`} className="sr-only">
                    Edit segment {index + 1}
                  </label>
                  <textarea
                    id={`segment-${segment.id}`}
                    lang="si"
                    value={segment.text}
                    onChange={(e) => editSegment(segment.id, e.target.value)}
                    disabled={transcript.status === 'FINALIZED'}
                    rows={Math.max(2, Math.ceil(segment.text.length / 55))}
                  />
                </div>
              </article>
            )
          })}
        </section>
        <aside className="media-panel">
          {!compact && (
            <div className="stats-card">
              <h3>
                <Icon name="file" size={15} /> Transcript data
              </h3>
              <div className="stats-card__grid">
                <div>
                  <p>Words</p>
                  <p>{stats.words.toLocaleString()}</p>
                </div>
                <div>
                  <p>Duration</p>
                  <p>{stats.duration}</p>
                </div>
              </div>
              <div className="stats-card__reading">
                <span>Reading time</span>
                <strong>~{stats.readingMinutes} min</strong>
              </div>
            </div>
          )}
          {compact && (
            <div className="media-preview">
              {playbackUrl ? (
                transcript.mediaType?.startsWith('video/') ? (
                  <video ref={player} controls src={playbackUrl} />
                ) : (
                  <audio ref={player} controls src={playbackUrl} />
                )
              ) : (
                <div className="media-placeholder">
                  <span className="sound-bars" aria-hidden="true">
                    {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].map((item) => (
                      <i key={item} />
                    ))}
                  </span>
                  <Icon name={mediaError ? 'alert' : 'play'} size={22} />
                  <small>
                    {mediaError ||
                      (transcript.mediaUrl ? 'Loading recording…' : 'No recording is attached')}
                  </small>
                </div>
              )}
            </div>
          )}
          {!compact && (
            <div className="search-replace">
              <h3>
                <Icon name="search" size={15} /> Smart search
              </h3>
              <label htmlFor="search-word">Search word</label>
              <input
                id="search-word"
                value={searchWord}
                onChange={(e) => setSearchWord(e.target.value)}
                placeholder="e.g. baryon"
                disabled={transcript.status === 'FINALIZED'}
              />
              <label htmlFor="replace-word">Replace with</label>
              <input
                id="replace-word"
                value={replaceWord}
                onChange={(e) => setReplaceWord(e.target.value)}
                placeholder="e.g. proton"
                disabled={transcript.status === 'FINALIZED'}
              />
              <button
                type="button"
                className="button button--primary button--full"
                onClick={executeReplace}
                disabled={!searchWord.trim() || transcript.status === 'FINALIZED'}
              >
                Execute replace
              </button>
            </div>
          )}
          <div className="confidence-control">
            <h3>
              <Icon name="alert" size={15} /> Confidence threshold
            </h3>
            <label htmlFor={`threshold-${transcript.id}`}>
              <span>
                <small>Flag words below {Math.round(confidenceThreshold * 100)}%</small>
              </span>
              <output>{Math.round(confidenceThreshold * 100)}%</output>
            </label>
            <input
              id={`threshold-${transcript.id}`}
              type="range"
              min="0.5"
              max="0.95"
              step="0.05"
              value={confidenceThreshold}
              onChange={(e) => updatePreference('confidenceThreshold', Number(e.target.value))}
            />
            <p>
              <span className="low-confidence low-confidence--sample">
                <span className="confidence-mark">?</span>word
              </span>{' '}
              Needs review
            </p>
          </div>
        </aside>
      </div>
      {compact && (
        <div className="compact-editor-actions">
          <span>{dirty ? '● Unsaved answer edits' : 'Answer transcript saved'}</span>
          <button className="button button--secondary" disabled={!dirty || saving} onClick={save}>
            Save correction
          </button>
          {transcript.status !== 'FINALIZED' && (
            <button
              className="button button--primary"
              disabled={saving}
              onClick={() => setConfirmFinalize(true)}
            >
              Use this answer
            </button>
          )}
        </div>
      )}
      <ConfirmDialog
        open={confirmFinalize}
        title={compact ? 'Use this spoken answer?' : 'Finalize this transcript?'}
        message="You will not be able to edit it after finalizing. Make sure all low-confidence words have been reviewed."
        confirmLabel={compact ? 'Use answer' : 'Finalize transcript'}
        onCancel={() => setConfirmFinalize(false)}
        onConfirm={finalize}
      />
    </div>
  )
}
