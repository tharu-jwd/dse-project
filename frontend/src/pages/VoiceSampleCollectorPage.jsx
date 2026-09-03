import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api'
import Icon from '../components/Icon'
import { Alert, PageHeader } from '../components/UI'
import { useLanguage } from '../contexts/LanguageContext'

/**
 * Dev-only tool: step 1 of the voice-command embedding work needs real
 * recordings of the six command phrases to answer "do same-phrase
 * embeddings cluster?" - this records them straight from the browser
 * instead of requiring a manual file transfer. Not the real per-student
 * enrollment UI (that depends on what this data shows).
 */
export default function VoiceSampleCollectorPage() {
  const { t } = useLanguage()
  const [searchParams, setSearchParams] = useSearchParams()
  const lang = searchParams.get('lang') === 'en' ? 'en' : 'si'
  const [commands, setCommands] = useState(null)
  const [error, setError] = useState('')
  const [activeId, setActiveId] = useState(null)
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)

  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)
  const intervalRef = useRef(null)

  const load = () => {
    setCommands(null)
    api
      .getVoiceSampleProgress(lang)
      .then((data) => setCommands(data.commands))
      .catch((cause) => setError(cause.message))
  }
  useEffect(load, [lang])
  useEffect(
    () => () => {
      window.clearInterval(intervalRef.current)
      streamRef.current?.getTracks().forEach((track) => track.stop())
    },
    [],
  )

  const start = async (commandId) => {
    setError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      chunksRef.current = []
      const recorder = new MediaRecorder(stream)
      recorderRef.current = recorder
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunksRef.current.push(event.data)
      }
      recorder.onstop = async () => {
        streamRef.current?.getTracks().forEach((track) => track.stop())
        window.clearInterval(intervalRef.current)
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
        setRecording(false)
        try {
          const result = await api.uploadVoiceSample(commandId, blob, lang)
          setCommands((prev) =>
            prev.map((command) =>
              command.id === commandId ? { ...command, count: result.counts[commandId] } : command,
            ),
          )
        } catch (cause) {
          setError(cause.message)
        }
      }
      recorder.start()
      setActiveId(commandId)
      setRecording(true)
      setSeconds(0)
      intervalRef.current = window.setInterval(() => setSeconds((value) => value + 1), 1000)
    } catch {
      setError(t('enroll.micDenied'))
    }
  }

  const stop = () => {
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
  }

  const reRecord = async (commandId) => {
    setError('')
    try {
      const result = await api.deleteVoiceSamples(commandId, lang)
      setCommands((prev) =>
        prev.map((command) =>
          command.id === commandId ? { ...command, count: result.counts[commandId] } : command,
        ),
      )
    } catch (cause) {
      setError(cause.message)
    }
  }

  return (
    <div className="page page--narrow">
      <PageHeader
        eyebrow={t('collector.eyebrow')}
        title={t('collector.title')}
        description={lang === 'en' ? t('collector.descriptionEn') : t('collector.descriptionSi')}
      />
      <div className="view-toggle" role="group" aria-label={t('collector.sampleLanguage')} style={{ marginBottom: 18 }}>
        <button
          type="button"
          aria-pressed={lang === 'si'}
          onClick={() => setSearchParams(lang === 'si' ? {} : { lang: 'si' })}
        >
          {t('collector.sinhalaCurrent')}
        </button>
        <button
          type="button"
          aria-pressed={lang === 'en'}
          onClick={() => setSearchParams({ lang: 'en' })}
        >
          {t('collector.englishTrial')}
        </button>
      </div>
      {error && <Alert>{error}</Alert>}
      {commands === null ? (
        <p className="muted">{t('collector.loading')}</p>
      ) : (
        <div className="manage-list">
          {commands.map((command) => (
            <article key={command.id}>
              <div>
                <div className="document-icon">
                  <Icon name="mic" />
                </div>
                <div>
                  <strong>{lang === 'en' ? command.id : command.phrase}</strong>
                  <small>
                    {lang === 'en' ? t('collector.yourEnglishWord') : command.id} ·{' '}
                    {t('collector.sampleCount', command.count)}
                  </small>
                </div>
              </div>
              {activeId === command.id && recording ? (
                <button type="button" className="button button--danger button--small" onClick={stop}>
                  <Icon name="stop" size={15} /> {t('collector.stopSeconds', seconds)}
                </button>
              ) : (
                <div className="row-actions">
                  {command.count > 0 && (
                    <button
                      type="button"
                      className="icon-button icon-button--danger"
                      title={t('collector.deleteAllSamples', command.id)}
                      onClick={() => reRecord(command.id)}
                    >
                      <Icon name="trash" size={16} />
                    </button>
                  )}
                  <button
                    type="button"
                    className="button button--secondary button--small"
                    disabled={recording}
                    onClick={() => start(command.id)}
                  >
                    <Icon name="mic" size={15} /> {t('collector.record')}
                  </button>
                </div>
              )}
            </article>
          ))}
        </div>
      )}
    </div>
  )
}
