import { createContext, useCallback, useContext, useMemo, useState } from 'react'

const ToastContext = createContext(null)

export function ToastProvider({ children }) {
  const [toast, setToast] = useState(null)
  const showToast = useCallback((message, type = 'success') => {
    const id = Date.now(); setToast({ id, message, type })
    window.setTimeout(() => setToast((item) => item?.id === id ? null : item), 3800)
  }, [])
  const value = useMemo(() => ({ showToast }), [showToast])
  return (
    <ToastContext.Provider value={value}>
      {children}
      {toast && <div className={`toast toast--${toast.type}`} role="status"><span aria-hidden="true">{toast.type === 'success' ? '✓' : '!'}</span>{toast.message}<button className="icon-button" onClick={() => setToast(null)} aria-label="Dismiss notification">×</button></div>}
    </ToastContext.Provider>
  )
}

export function useToast() { return useContext(ToastContext) }
