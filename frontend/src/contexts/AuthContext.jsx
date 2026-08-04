import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { api } from '../api'

const AuthContext = createContext(null)

function readUser() {
  try {
    return JSON.parse(localStorage.getItem('sinhaspeech_user'))
  } catch {
    return null
  }
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(readUser)

  useEffect(() => {
    const expire = () => setUser(null)
    window.addEventListener('sinhaspeech-session-expired', expire)
    return () => window.removeEventListener('sinhaspeech-session-expired', expire)
  }, [])

  const value = useMemo(
    () => ({
      user,
      async login(email, password) {
        const session = await api.login(email, password)
        localStorage.setItem('sinhaspeech_token', session.token)
        localStorage.setItem('sinhaspeech_user', JSON.stringify(session.user))
        setUser(session.user)
        return session.user
      },
      async logout() {
        try {
          await api.logout()
        } finally {
          localStorage.removeItem('sinhaspeech_token')
          localStorage.removeItem('sinhaspeech_user')
          setUser(null)
        }
      },
    }),
    [user],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  return useContext(AuthContext)
}
