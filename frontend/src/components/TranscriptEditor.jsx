import { useEffect, useMemo, useRef, useState } from 'react'
import { api, downloadBlob } from '../api'
import { useAccessibility } from '../contexts/AccessibilityContext'
import { useLanguage } from '../contexts/LanguageContext'
import { useToast } from '../contexts/ToastContext'
import useVoiceCommands from '../hooks/useVoiceCommands'
import Icon from './Icon'
import { Alert, ConfirmDialog, StatusBadge } from './UI'
import VoiceMeter from './VoiceMeter'

const formatTime = (seconds) =>
  `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`

function highlightMatches(text, term) {
  if (!term?.trim()) return text
  const pattern = new RegExp(`(${term.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
  const parts = text.split(pattern)
  if (parts.length === 1) return text
  return parts.map((part, index) =>
    pattern.test(part) && part.toLowerCase() === term.trim().toLowerCase() ? (
      <mark className="search-match" key={index}>
        {part}
      </mark>
    ) : (
      <span key={index}>{part}</span>
    ),
  )
}

export function ConfidenceText({ segment, threshold, highlightTerm, showConfidence = true }) {
  if (!segment.words?.length) {
    // A confidence of exactly 0 with no per-word data means no score was
    // ever recorded for this segment (e.g. live/note recordings) rather than
    // a genuinely bad transcription, so it should not be flagged as low.
    const flagged = showConfidence && segment.confidence > 0 && segment.confidence < threshold
    return (
      <span className={flagged ? 'low-confidence' : ''}>
        {highlightMatches(segment.text, highlightTerm)}
        {flagged && <span className="sr-only"> (low confidence)</span>}
      </span>
    )
  }
  return (
    <>
      {segment.words.map((word, index) => {
        // Same zero-confidence exemption as the segment fallback above -
        // a word with no real score (0) should never render as "low
        // confidence", only a word scored below the threshold should.
        const flagged = showConfidence && word.confidence > 0 && word.confidence < threshold
        return (
          <span key={`${word.text}-${index}`} className={flagged ? 'low-confidence' : ''}>
            {flagged && (
              <span className="confidence-mark" aria-hidden="true">
                ?
              </span>
            )}
            {highlightMatches(word.text, highlightTerm)}
            {index < segment.words.length - 1 ? ' ' : ''}
            {flagged && <span className="sr-only"> (low confidence)</span>}
          </span>
        )
      })}
    </>
  )
}

export default function TranscriptEditor({
  initialTranscript,
  onTranscriptChange,
  compact = false,
}) {
  const { t } = useLanguage()
  const [transcript, setTranscript] = useState(initialTranscript)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [confirmFinalize, setConfirmFinalize] = useState(false)
  const { confidenceThreshold, interactionMode, updatePreference } = useAccessibility()
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
  const voice = useVoiceCommands({
    onCommand: (command) => {
      if (transcript.status === 'FINALIZED') return
      // The confirm dialog takes priority over the normal per-command
      // routing below: once it's open, "submit" said again confirms it
      // and "cancel" dismisses it, rather than either doing nothing or
      // (for "submit") just re-opening the same dialog.
      if (confirmFinalize) {
        if (command === 'submit') {
          setConfirmFinalize(false)
          finalize()
        } else if (command === 'cancel') {
          setConfirmFinalize(false)
          showCommandFeedback(t('editor.finalizeCancelled'))
        }
        return
      }
      if (command === 'save') {
        if (!dirty) {
          showCommandFeedback(t('editor.nothingToSave'))
          return
        }
        showCommandFeedback(t('editor.saving'))
        save()
      } else if (command === 'submit') {
        showCommandFeedback(t('editor.openingFinalizeConfirmation'))
        setConfirmFinalize(true)
      }
    },
  })
  useEffect(() => {
    if (interactionMode !== 'command' && voice.isListening) voice.stop()
  }, [interactionMode]) // eslint-disable-line react-hooks/exhaustive-deps
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
            cause.message || t('editor.recordingLoadFailed')
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
      showToast(t('editor.changesSaved'))
    } catch (cause) {
      setError(cause.message || t('editor.changesSaveFailed'))
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
      showToast(t('editor.finalizedSuccessfully'))
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
      showToast(t('editor.exportDownloaded', format.toUpperCase()))
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
  // Confidence scores only ever exist for uploaded/recorded lectures (the
  // Whisper batch pipeline). Notes and quiz answers are either live-streamed
  // (never scored) or edited right in this same space before submission, so
  // the confidence UI is noise there - show it for lectures only.
  const showConfidence = transcript.type === 'LECTURE'
  const matchCount = useMemo(() => {
    const term = searchWord.trim().toLowerCase()
    if (!term) return 0
    return transcript.segments.reduce((sum, segment) => {
      const text = segment.text.toLowerCase()
      let count = 0
      let index = text.indexOf(term)
      while (index !== -1) {
        count += 1
        index = text.indexOf(term, index + term.length)
      }
      return sum + count
    }, 0)
  }, [searchWord, transcript.segments])
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
              {t('editor.transcriptTitle')}
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
              <span>{t('editor.segments', transcript.segments.length)}</span>
              {dirty && <span className="unsaved-dot">{t('editor.unsavedChanges')}</span>}
            </div>
          </div>
          <div className="editor-toolbar__actions">
            {interactionMode === 'command' && transcript.status !== 'FINALIZED' && (
              <div className="voice-command-toggle">
                <button
                  type="button"
                  className={`button button--icon ${voice.isListening ? 'is-recording' : ''}`}
                  onClick={voice.isListening ? voice.stop : voice.start}
                  disabled={voice.status === 'connecting' || voice.status === 'stopping'}
                  aria-label={
                    voice.isListening ? t('editor.stopVoiceCommands') : t('editor.listenVoiceCommands')
                  }
                  title={voice.isListening ? t('editor.stopVoiceCommands') : t('editor.sayCommand')}
                >
                  <Icon name={voice.isListening ? 'stop' : 'mic'} size={17} />
                </button>
                {voice.isListening && (
                  <VoiceMeter registerBar={voice.registerBar} active={voice.voiceDetected} compact />
                )}
              </div>
            )}
            {!compact && (
              <div className="export-toggle" role="group" aria-label={t('editor.exportFormat')}>
                {exportFormats.map((format) => (
                  <button
                    type="button"
                    key={format}
                    className={format === lastFormat ? 'active' : ''}
                    onClick={() => exportFile(format)}
                    disabled={dirty}
                    title={
                      dirty
                        ? t('editor.saveBeforeExporting')
                        : t('editor.exportFormatLabel', format.toUpperCase())
                    }
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
              {t('editor.saveDraft')}
            </button>
            {transcript.status !== 'FINALIZED' && (
              <button
                type="button"
                className="button button--primary"
                disabled={saving}
                onClick={() => setConfirmFinalize(true)}
              >
                {t('editor.finalize')}
              </button>
            )}
          </div>
        </div>
      )}
      {error && (
        <Alert>
          {error} {dirty && t('editor.unsavedEditsPreserved')}
        </Alert>
      )}
      {voice.error && <Alert>{voice.error}</Alert>}
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
                aria-label={isPlaying ? t('editor.pause') : t('editor.play')}
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
                  aria-label={t('editor.seekRecording')}
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
                  (transcript.mediaUrl ? t('editor.loadingRecording') : t('editor.noRecordingAttached'))}
              </span>
            </>
          )}
        </div>
      )}
      <div className="editor-layout">
        <section className="segments" aria-label={t('editor.segmentsLabel')}>
          {transcript.segments.map((segment, index) => {
            const isLow =
              showConfidence && segment.confidence > 0 && segment.confidence < confidenceThreshold
            return (
              <article className={`segment ${isLow ? 'segment--review' : ''}`} key={segment.id}>
                <div className="segment__gutter">
                  <button
                    type="button"
                    className="timestamp"
                    onClick={() => seek(segment.startTime)}
                    aria-label={t('editor.playFrom', formatTime(segment.startTime))}
                    title={t('editor.playFrom', formatTime(segment.startTime))}
                  >
                    {formatTime(segment.startTime)}
                  </button>
                  {isLow && (
                    <span
                      className="confidence-flag"
                      title={t('editor.confidencePercentReview', Math.round(segment.confidence * 100))}
                    >
                      <Icon name="alert" size={13} />
                      <span className="sr-only">
                        {t('editor.confidencePercentReviewSr', Math.round(segment.confidence * 100))}
                      </span>
                    </span>
                  )}
                  <span className="sr-only">{t('editor.segmentN', index + 1)}</span>
                </div>
                <div className="segment__body">
                  <div className="confidence-preview" lang="si">
                    <ConfidenceText
                      segment={segment}
                      threshold={confidenceThreshold}
                      highlightTerm={searchWord}
                      showConfidence={showConfidence}
                    />
                  </div>
                  <label htmlFor={`segment-${segment.id}`} className="sr-only">
                    {t('editor.editSegmentN', index + 1)}
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
                <Icon name="file" size={15} /> {t('editor.transcriptData')}
              </h3>
              <div className="stats-card__grid">
                <div>
                  <p>{t('editor.words')}</p>
                  <p>{stats.words.toLocaleString()}</p>
                </div>
                <div>
                  <p>{t('editor.duration')}</p>
                  <p>{stats.duration}</p>
                </div>
              </div>
              <div className="stats-card__reading">
                <span>{t('editor.readingTime')}</span>
                <strong>{t('editor.readingMinutes', stats.readingMinutes)}</strong>
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
                      (transcript.mediaUrl ? t('editor.loadingRecording') : t('editor.noRecordingAttached'))}
                  </small>
                </div>
              )}
            </div>
          )}
          {!compact && (
            <div className="search-replace">
              <h3>
                <Icon name="search" size={15} /> {t('editor.smartSearch')}
              </h3>
              <label htmlFor="search-word">{t('editor.searchWord')}</label>
              <input
                id="search-word"
                value={searchWord}
                onChange={(e) => setSearchWord(e.target.value)}
                placeholder={t('editor.searchPlaceholder')}
                disabled={transcript.status === 'FINALIZED'}
              />
              {searchWord.trim() && (
                <p className="search-match-count">
                  {matchCount ? t('editor.matchesFound', matchCount) : t('editor.noMatchesFound')}
                </p>
              )}
              <label htmlFor="replace-word">{t('editor.replaceWith')}</label>
              <input
                id="replace-word"
                value={replaceWord}
                onChange={(e) => setReplaceWord(e.target.value)}
                placeholder={t('editor.replacePlaceholder')}
                disabled={transcript.status === 'FINALIZED'}
              />
              <button
                type="button"
                className="button button--primary button--full"
                onClick={executeReplace}
                disabled={!searchWord.trim() || transcript.status === 'FINALIZED'}
              >
                {t('editor.executeReplace')}
              </button>
            </div>
          )}
          {!compact && showConfidence && (
            <div className="confidence-control">
              <h3>
                <Icon name="alert" size={15} /> {t('editor.confidenceThreshold')}
              </h3>
              <label htmlFor={`threshold-${transcript.id}`}>
                <span>
                  <small>{t('editor.flagWordsBelow', Math.round(confidenceThreshold * 100))}</small>
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
                {t('editor.needsReview')}
              </p>
            </div>
          )}
        </aside>
      </div>
      {compact && (
        <div className="compact-editor-actions">
          <span>{dirty ? t('editor.unsavedAnswerEdits') : t('editor.answerTranscriptSaved')}</span>
          <button
            className="button button--primary"
            disabled={saving}
            onClick={() => (dirty ? save() : onTranscriptChange?.(transcript))}
          >
            {saving ? (
              <span className="spinner spinner--small" />
            ) : (
              <Icon name="check" size={17} />
            )}{' '}
            {t('editor.saveAnswer')}
          </button>
        </div>
      )}
      {!compact && (
        <ConfirmDialog
          open={confirmFinalize}
          title={t('editor.finalizeTranscriptTitle')}
          message={t('editor.finalizeMessage')}
          confirmLabel={t('editor.finalizeTranscript')}
          onCancel={() => setConfirmFinalize(false)}
          onConfirm={finalize}
        />
      )}
    </div>
  )
}
