import { useEffect, useRef, useState } from 'react'
import { api, downloadBlob } from '../api'
import { useAccessibility } from '../contexts/AccessibilityContext'
import { useToast } from '../contexts/ToastContext'
import Icon from './Icon'
import { Alert, ConfirmDialog, StatusBadge } from './UI'

const formatTime = (seconds) => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`

export function ConfidenceText({ segment, threshold }) {
  if (!segment.words?.length) return <span className={segment.confidence < threshold ? 'low-confidence' : ''}>{segment.text}{segment.confidence < threshold && <span className="sr-only"> (low confidence)</span>}</span>
  return <>{segment.words.map((word, index) => <span key={`${word.text}-${index}`} className={word.confidence < threshold ? 'low-confidence' : ''}>{word.confidence < threshold && <span className="confidence-mark" aria-hidden="true">?</span>}{word.text}{index < segment.words.length - 1 ? ' ' : ''}{word.confidence < threshold && <span className="sr-only"> (low confidence)</span>}</span>)}</>
}

export default function TranscriptEditor({ initialTranscript, onTranscriptChange, compact = false }) {
  const [transcript, setTranscript] = useState(initialTranscript); const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false); const [error, setError] = useState(''); const [confirmFinalize, setConfirmFinalize] = useState(false)
  const { confidenceThreshold, updatePreference } = useAccessibility(); const { showToast } = useToast(); const player = useRef(null)
  useEffect(() => setTranscript(initialTranscript), [initialTranscript])

  const editSegment = (id, text) => {
    setTranscript((current) => ({ ...current, segments: current.segments.map((segment) => segment.id === id ? { ...segment, text } : segment) }))
    setDirty(true); setError('')
  }
  const save = async () => {
    setSaving(true); setError('')
    try { const saved = await api.updateTranscript(transcript.id, { title: transcript.title, segments: transcript.segments }); setTranscript(saved); setDirty(false); onTranscriptChange?.(saved); showToast('Transcript changes saved.') }
    catch (cause) { setError(cause.message || 'Changes could not be saved. Your edits are still here.') }
    finally { setSaving(false) }
  }
  const finalize = async () => {
    setConfirmFinalize(false); setSaving(true); setError('')
    try { if (dirty) await api.updateTranscript(transcript.id, { title: transcript.title, segments: transcript.segments }); const result = await api.finalizeTranscript(transcript.id); setTranscript(result); setDirty(false); onTranscriptChange?.(result); showToast('Transcript finalized successfully.') }
    catch (cause) { setError(cause.message); } finally { setSaving(false) }
  }
  const exportFile = async (format) => {
    try { const blob = await api.exportTranscript(transcript.id, format); downloadBlob(blob, `${transcript.title.replace(/[^a-zA-Z0-9\u0D80-\u0DFF]+/g, '-')}.${format}`); showToast(`${format.toUpperCase()} export downloaded.`) }
    catch (cause) { setError(cause.message) }
  }
  const seek = (seconds) => { if (player.current && transcript.mediaUrl) { player.current.currentTime = seconds; player.current.play().catch(() => {}) } }
  const exportFormats = transcript.type === 'LECTURE' ? ['txt', 'docx', 'pdf'] : ['txt', 'docx']

  return <div className={`transcript-editor ${compact ? 'transcript-editor--compact' : ''}`}>
    {!compact && <div className="editor-toolbar"><div><label htmlFor="transcript-title" className="sr-only">Transcript title</label><input id="transcript-title" className="title-input" value={transcript.title} onChange={(e) => { setTranscript((current) => ({ ...current, title: e.target.value })); setDirty(true) }} /><div className="editor-meta"><StatusBadge status={transcript.status} /><span>{transcript.segments.length} segments</span>{dirty && <span className="unsaved-dot">● Unsaved changes</span>}</div></div><div className="editor-toolbar__actions"><button type="button" className="button button--secondary" disabled={saving || !dirty || transcript.status === 'FINALIZED'} onClick={save}>{saving ? <span className="spinner spinner--small" /> : <Icon name="check" size={17} />} Save changes</button>{transcript.status !== 'FINALIZED' && <button type="button" className="button button--primary" disabled={saving} onClick={() => setConfirmFinalize(true)}>Finalize</button>}</div></div>}
    {error && <Alert>{error} {dirty && 'Your unsaved edits have been preserved.'}</Alert>}
    <div className="editor-layout">
      <aside className="media-panel"><div className="media-preview">{transcript.mediaUrl ? (transcript.mediaType?.startsWith('video/') ? <video ref={player} controls src={transcript.mediaUrl} /> : <audio ref={player} controls src={transcript.mediaUrl} />) : <div className="media-placeholder"><span className="sound-bars" aria-hidden="true">{[1,2,3,4,5,6,7,8,9,10,11].map((item) => <i key={item} />)}</span><Icon name="play" size={22} /><small>Media preview unavailable in sample data</small></div>}</div><div className="confidence-control"><label htmlFor={`threshold-${transcript.id}`}><span><strong>Confidence threshold</strong><small>Flag words below {Math.round(confidenceThreshold * 100)}%</small></span><output>{Math.round(confidenceThreshold * 100)}%</output></label><input id={`threshold-${transcript.id}`} type="range" min="0.5" max="0.95" step="0.05" value={confidenceThreshold} onChange={(e) => updatePreference('confidenceThreshold', Number(e.target.value))} /><p><span className="low-confidence low-confidence--sample"><span className="confidence-mark">?</span>word</span> Needs review</p></div>{!compact && <div className="export-box"><strong>Export transcript</strong><div>{exportFormats.map((format) => <button type="button" key={format} className="button button--secondary button--small" onClick={() => exportFile(format)} disabled={dirty} title={dirty ? 'Save changes before exporting' : ''}><Icon name="download" size={15} />{format.toUpperCase()}</button>)}</div></div>}</aside>
      <section className="segments" aria-label="Transcript segments">{transcript.segments.map((segment, index) => <article className={`segment ${segment.confidence < confidenceThreshold ? 'segment--review' : ''}`} key={segment.id}><div className="segment__top"><button type="button" className="timestamp" onClick={() => seek(segment.startTime)} aria-label={`Play from ${formatTime(segment.startTime)}`}><Icon name="play" size={13} /> {formatTime(segment.startTime)}</button><span>Segment {index + 1}</span><span className={`confidence-score ${segment.confidence < confidenceThreshold ? 'is-low' : ''}`}>{segment.confidence < confidenceThreshold && <Icon name="alert" size={14} />} {Math.round(segment.confidence * 100)}% confidence</span></div><div className="confidence-preview" lang="si"><ConfidenceText segment={segment} threshold={confidenceThreshold} /></div><label htmlFor={`segment-${segment.id}`}>Edit segment <span className="sr-only">{index + 1}</span></label><textarea id={`segment-${segment.id}`} lang="si" value={segment.text} onChange={(e) => editSegment(segment.id, e.target.value)} disabled={transcript.status === 'FINALIZED'} rows={Math.max(2, Math.ceil(segment.text.length / 55))} /></article>)}</section>
    </div>
    {compact && <div className="compact-editor-actions"><span>{dirty ? '● Unsaved answer edits' : 'Answer transcript saved'}</span><button className="button button--secondary" disabled={!dirty || saving} onClick={save}>Save correction</button>{transcript.status !== 'FINALIZED' && <button className="button button--primary" disabled={saving} onClick={() => setConfirmFinalize(true)}>Use this answer</button>}</div>}
    <ConfirmDialog open={confirmFinalize} title={compact ? 'Use this spoken answer?' : 'Finalize this transcript?'} message="You will not be able to edit it after finalizing. Make sure all low-confidence words have been reviewed." confirmLabel={compact ? 'Use answer' : 'Finalize transcript'} onCancel={() => setConfirmFinalize(false)} onConfirm={finalize} />
  </div>
}
