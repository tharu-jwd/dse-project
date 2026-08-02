import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api'

export default function useTranscriptionJob() {
  const [job, setJob] = useState(null); const timer = useRef(null); const active = useRef(true)
  useEffect(() => () => { active.current = false; window.clearTimeout(timer.current) }, [])

  const poll = useCallback(async (jobId) => {
    try {
      const next = await api.getJob(jobId)
      if (!active.current) return
      setJob(next)
      if (next.status === 'UPLOADING' || next.status === 'PROCESSING') timer.current = window.setTimeout(() => poll(jobId), 800)
    } catch (error) { if (active.current) setJob({ jobId, status: 'FAILED', message: error.message }) }
  }, [])

  const start = useCallback(async (formData) => {
    window.clearTimeout(timer.current); setJob({ status: 'UPLOADING' })
    try { const created = await api.createTranscription(formData); if (active.current) { setJob(created); timer.current = window.setTimeout(() => poll(created.jobId), 500) } }
    catch (error) { if (active.current) setJob({ status: 'FAILED', message: error.message }) }
  }, [poll])
  const reset = useCallback(() => { window.clearTimeout(timer.current); setJob(null) }, [])
  return { job, start, reset }
}
