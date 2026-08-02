import { DEMO_USERS, initialQuizzes, initialSubmissions, initialTranscripts, sampleSegments } from './mockData'

export const USE_MOCK_API = String(import.meta.env.VITE_USE_MOCK_API ?? 'true') === 'true'
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const pause = (ms = 450) => new Promise((resolve) => setTimeout(resolve, ms))
const clone = (value) => structuredClone(value)
let transcripts = clone(initialTranscripts)
let quizzes = clone(initialQuizzes)
let submissions = clone(initialSubmissions)
const jobs = new Map()

function currentUser() {
  try { return JSON.parse(localStorage.getItem('sinhaspeech_user')) } catch { return null }
}

function token() { return localStorage.getItem('sinhaspeech_token') }

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {})
  if (token()) headers.set('Authorization', `Bearer ${token()}`)
  if (options.body && !(options.body instanceof FormData)) headers.set('Content-Type', 'application/json')
  let response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers })
  } catch {
    throw Object.assign(new Error('Unable to reach the server. Check your connection and try again.'), { code: 'NETWORK_ERROR' })
  }
  if (response.status === 401) {
    localStorage.removeItem('sinhaspeech_token')
    localStorage.removeItem('sinhaspeech_user')
    window.dispatchEvent(new Event('sinhaspeech-session-expired'))
    throw new Error('Your session has expired. Please sign in again.')
  }
  if (!response.ok) {
    const data = await response.json().catch(() => ({}))
    throw new Error(data.message || 'The request could not be completed.')
  }
  const contentType = response.headers.get('content-type') || ''
  return contentType.includes('application/json') ? response.json() : response.blob()
}

function makeTranscript({ title, type, file }) {
  const id = `transcript-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
  return {
    id, ownerId: currentUser()?.id, title: title || 'Untitled transcript', type,
    status: 'DRAFT', date: new Date().toISOString(),
    mediaUrl: file ? URL.createObjectURL(file) : '', mediaType: file?.type || '', segments: sampleSegments(),
  }
}

export const api = {
  async login(email, password) {
    if (!USE_MOCK_API) return request('/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) })
    await pause(700)
    const user = Object.values(DEMO_USERS).find((item) => item.email === email && item.password === password)
    if (!user) throw Object.assign(new Error('The email or password is incorrect.'), { code: 'INVALID_CREDENTIALS' })
    const { password: _password, ...safeUser } = user
    return { token: `mock-jwt-${user.role.toLowerCase()}`, user: safeUser }
  },
  async logout() {
    if (!USE_MOCK_API) await request('/auth/logout', { method: 'POST' })
    await pause(150)
  },
  async createTranscription(formData) {
    if (!USE_MOCK_API) return request('/transcriptions', { method: 'POST', body: formData })
    await pause(700)
    const file = formData.get('file')
    const jobId = `job-${Date.now()}`
    jobs.set(jobId, {
      jobId, status: 'UPLOADING', polls: 0,
      shouldFail: String(file?.name || '').toLowerCase().includes('fail'),
      transcript: makeTranscript({ title: formData.get('title'), type: formData.get('type') || 'LECTURE', file }),
    })
    return { jobId, status: 'UPLOADING' }
  },
  async getJob(jobId) {
    if (!USE_MOCK_API) return request(`/transcriptions/jobs/${jobId}`)
    await pause(350)
    const job = jobs.get(jobId)
    if (!job) throw new Error('Transcription job was not found.')
    job.polls += 1
    if (job.polls === 1) job.status = 'PROCESSING'
    if (job.polls >= 3) {
      job.status = job.shouldFail ? 'FAILED' : 'COMPLETED'
      if (job.status === 'COMPLETED' && !transcripts.some((item) => item.id === job.transcript.id)) transcripts.unshift(job.transcript)
    }
    return { jobId, status: job.status, transcriptId: job.status === 'COMPLETED' ? job.transcript.id : undefined, message: job.status === 'FAILED' ? 'The audio could not be transcribed. Please check the file and retry.' : undefined }
  },
  async getTranscripts() {
    if (!USE_MOCK_API) return request('/transcripts')
    await pause(); const user = currentUser()
    return clone(user?.role === 'TEACHER' ? transcripts.filter((t) => t.ownerId === user.id || t.type === 'LECTURE') : transcripts.filter((t) => t.ownerId === user?.id))
  },
  async getTranscript(id) {
    if (!USE_MOCK_API) return request(`/transcripts/${id}`)
    await pause(300); const item = transcripts.find((t) => t.id === id)
    if (!item) throw new Error('Transcript was not found.')
    return clone(item)
  },
  async updateTranscript(id, updates) {
    if (!USE_MOCK_API) return request(`/transcripts/${id}`, { method: 'PATCH', body: JSON.stringify(updates) })
    await pause(500); const index = transcripts.findIndex((t) => t.id === id)
    if (index < 0) throw new Error('Transcript was not found.')
    transcripts[index] = { ...transcripts[index], ...clone(updates) }
    return clone(transcripts[index])
  },
  async finalizeTranscript(id) {
    if (!USE_MOCK_API) return request(`/transcripts/${id}/finalize`, { method: 'POST' })
    return api.updateTranscript(id, { status: 'FINALIZED' })
  },
  async exportTranscript(id, format) {
    if (!USE_MOCK_API) return request(`/transcripts/${id}/export?format=${format}`)
    await pause(400); const item = transcripts.find((t) => t.id === id)
    if (!item) throw new Error('Transcript was not found.')
    return new Blob([`${item.title}\n\n${item.segments.map((s) => s.text).join('\n')}`], { type: format === 'txt' ? 'text/plain;charset=utf-8' : 'application/octet-stream' })
  },
  async getQuizzes({ teacher = false } = {}) {
    if (!USE_MOCK_API) return request('/quizzes')
    await pause(); return clone(teacher ? quizzes : quizzes.filter((q) => q.status === 'PUBLISHED'))
  },
  async getQuiz(id) {
    if (!USE_MOCK_API) return request(`/quizzes/${id}`)
    await pause(300); const quiz = quizzes.find((q) => q.id === id)
    if (!quiz) throw new Error('Quiz was not found.')
    return clone(quiz)
  },
  async saveQuiz(quiz) {
    const payload = { ...quiz, id: quiz.id || `quiz-${Date.now()}` }
    if (!USE_MOCK_API) return request(quiz.id ? `/quizzes/${quiz.id}` : '/quizzes', { method: quiz.id ? 'PATCH' : 'POST', body: JSON.stringify(payload) })
    await pause(500); const index = quizzes.findIndex((q) => q.id === payload.id)
    if (index >= 0) quizzes[index] = clone(payload); else quizzes.unshift(clone(payload))
    return clone(payload)
  },
  async publishQuiz(id) {
    if (!USE_MOCK_API) return request(`/quizzes/${id}/publish`, { method: 'POST' })
    const quiz = quizzes.find((q) => q.id === id); return api.saveQuiz({ ...quiz, status: 'PUBLISHED' })
  },
  async submitQuiz(id, answers) {
    if (!USE_MOCK_API) return request(`/quizzes/${id}/submit`, { method: 'POST', body: JSON.stringify({ answers }) })
    await pause(600); const quiz = quizzes.find((q) => q.id === id)
    if (quiz) quiz.submissionStatus = 'SUBMITTED'
    return { status: 'SUBMITTED', submittedAt: new Date().toISOString() }
  },
  async getSubmissions() {
    if (!USE_MOCK_API) return request('/submissions')
    await pause(); return clone(submissions)
  },
  async getSubmission(id) {
    if (!USE_MOCK_API) return request(`/submissions/${id}`)
    await pause(300); const result = submissions.find((s) => s.id === id)
    if (!result) throw new Error('Submission was not found.')
    return clone(result)
  },
  async reviewSubmission(id, review) {
    if (!USE_MOCK_API) return request(`/submissions/${id}/review`, { method: 'PATCH', body: JSON.stringify(review) })
    await pause(500); const item = submissions.find((s) => s.id === id)
    Object.assign(item, review, { status: 'REVIEWED' }); return clone(item)
  },
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a'); link.href = url; link.download = filename
  document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(url)
}
