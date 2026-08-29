import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { api, downloadBlob } from '../api'
import editorBackground from '../assets/1.jpg'
import Icon from '../components/Icon'
import TranscriptEditor from '../components/TranscriptEditor'
import { Alert, ConfirmDialog, EmptyState, Loading, PageHeader, StatusBadge } from '../components/UI'
import { useToast } from '../contexts/ToastContext'

const typeLabel = { LECTURE: 'Lecture', NOTE: 'Study note', QUIZ_ANSWER: 'Quiz answer' }

export function TranscriptLibraryPage() {
  const [searchParams] = useSearchParams()
  const [items, setItems] = useState(null)
  const [error, setError] = useState('')
  const [search, setSearch] = useState(searchParams.get('q') || '')
  const [type, setType] = useState('ALL')
  const [status, setStatus] = useState('ALL')
  const [deleteTarget, setDeleteTarget] = useState(null)
  const [isDeleting, setIsDeleting] = useState(false)
  const { showToast } = useToast()
  const load = () => {
    setItems(null)
    setError('')
    api
      .getTranscripts()
      .then(setItems)
      .catch((cause) => setError(cause.message))
  }
  useEffect(load, [])
  const filtered = useMemo(
    () =>
      (items || []).filter(
        (item) =>
          item.title.toLowerCase().includes(search.toLowerCase()) &&
          (type === 'ALL' || item.type === type) &&
          (status === 'ALL' || item.status === status),
      ),
    [items, search, type, status],
  )
  const exportItem = async (item) => {
    try {
      const blob = await api.exportTranscript(item.id, 'txt')
      downloadBlob(blob, `${item.title}.txt`)
      showToast('Transcript downloaded.')
    } catch (cause) {
      setError(cause.message)
    }
  }
  const deleteItem = async () => {
    if (isDeleting) return
    const target = deleteTarget
    setIsDeleting(true)
    try {
      try {
        await api.deleteTranscript(target.id)
      } catch (cause) {
        if (!/not found/i.test(cause.message)) throw cause
      }
      setItems((prev) => (prev || []).filter((entry) => entry.id !== target.id))
      showToast('Transcript deleted.')
      setDeleteTarget(null)
    } catch (cause) {
      setError(cause.message)
    } finally {
      setIsDeleting(false)
    }
  }
  return (
    <div className="page">
      <PageHeader
        eyebrow="Your saved work"
        title="Transcript library"
        description="Search, review and export your lecture captions, study notes and spoken answers."
      />
      <section className="library-tools" aria-label="Transcript filters">
        <label className="search-field">
          <span className="sr-only">Search transcripts</span>
          <Icon name="search" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by title…"
          />
        </label>
        <label>
          <span className="sr-only">Filter by type</span>
          <select value={type} onChange={(e) => setType(e.target.value)}>
            <option value="ALL">All types</option>
            <option value="LECTURE">Lectures</option>
            <option value="NOTE">Study notes</option>
            <option value="QUIZ_ANSWER">Quiz answers</option>
          </select>
        </label>
        <label>
          <span className="sr-only">Filter by status</span>
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="ALL">All statuses</option>
            <option value="DRAFT">Draft</option>
            <option value="FINALIZED">Finalized</option>
            <option value="PROCESSING">Processing</option>
            <option value="FAILED">Failed</option>
          </select>
        </label>
      </section>
      {error && (
        <Alert title="Could not load transcripts">
          {error}
          <div>
            <button className="button button--secondary button--small" onClick={load}>
              Try again
            </button>
          </div>
        </Alert>
      )}
      {items === null && !error ? (
        <Loading label="Loading your transcript library…" />
      ) : filtered.length ? (
        <div className="table-card">
          <table>
            <thead>
              <tr>
                <th>Transcript</th>
                <th>Type</th>
                <th>Date created</th>
                <th>Status</th>
                <th>
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((item) => (
                <tr key={item.id}>
                  <td>
                    <span className={`document-icon document-icon--${item.type.toLowerCase()}`}>
                      <Icon
                        name={
                          item.type === 'NOTE'
                            ? 'mic'
                            : item.type === 'QUIZ_ANSWER'
                              ? 'quiz'
                              : 'file'
                        }
                      />
                    </span>
                    <strong>{item.title}</strong>
                  </td>
                  <td>{typeLabel[item.type]}</td>
                  <td>
                    {new Intl.DateTimeFormat('en-GB', {
                      day: 'numeric',
                      month: 'short',
                      year: 'numeric',
                    }).format(new Date(item.date))}
                  </td>
                  <td>
                    <StatusBadge status={item.status} />
                  </td>
                  <td>
                    <div className="row-actions">
                      <Link
                        className="button button--secondary button--small"
                        to={`/transcripts/${item.id}`}
                      >
                        Open
                      </Link>
                      {item.status === 'FINALIZED' && (
                        <button
                          className="icon-button"
                          title="Export TXT"
                          aria-label={`Export ${item.title}`}
                          onClick={() => exportItem(item)}
                        >
                          <Icon name="download" size={18} />
                        </button>
                      )}
                      <button
                        className="icon-button"
                        title="Delete"
                        aria-label={`Delete ${item.title}`}
                        onClick={() => setDeleteTarget(item)}
                      >
                        <Icon name="trash" size={18} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <EmptyState
          icon="search"
          title={items?.length ? 'No matching transcripts' : 'Your library is ready'}
          message={
            items?.length
              ? 'Try changing your search or filters.'
              : 'Uploaded lectures, recorded notes and spoken answers will appear here.'
          }
          action={
            !items?.length && (
              <Link className="button button--primary" to="/lectures/new">
                Create a transcript
              </Link>
            )
          }
        />
      )}
      <ConfirmDialog
        open={Boolean(deleteTarget)}
        title="Delete transcript"
        message={`Are you sure you want to delete "${deleteTarget?.title}"? This cannot be undone.`}
        confirmLabel="Delete transcript"
        dangerous
        busy={isDeleting}
        onConfirm={deleteItem}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  )
}

export function TranscriptPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [item, setItem] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    api
      .getTranscript(id)
      .then((value) => active && setItem(value))
      .catch((cause) => active && setError(cause.message))
    return () => {
      active = false
    }
  }, [id])
  if (error)
    return (
      <div className="page page--narrow">
        <Alert title="Could not open transcript">{error}</Alert>
        <button className="button button--secondary" onClick={() => navigate('/transcripts')}>
          Back to library
        </button>
      </div>
    )
  if (!item)
    return (
      <div className="page">
        <Loading label="Opening transcript editor…" />
      </div>
    )
  return (
    <div
      className="page page--editor has-bg-image"
      style={{ backgroundImage: `url(${editorBackground})` }}
    >
      <PageHeader
        eyebrow={`${typeLabel[item.type]} transcript`}
        title="Review transcript"
        description="Correct uncertain words, save your changes and finalize when it is ready."
        back={
          <button className="back-link" onClick={() => navigate(-1)}>
            ← Back
          </button>
        }
      />
      <TranscriptEditor initialTranscript={item} onTranscriptChange={setItem} />
    </div>
  )
}
