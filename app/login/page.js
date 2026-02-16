"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const router = useRouter();

  const handleSubmit = async (e) => {
    e.preventDefault();

    const response = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });

    if (response.ok) {
      router.push("/");
      router.refresh();
    } else {
      setError("Incorrect password");
      setPassword("");
    }
  };

  return (
    <div style={{
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      minHeight: "100vh",
      backgroundColor: "#0a0a0f",
      fontFamily: "'DM Sans','Helvetica Neue',sans-serif",
      padding: "16px"
    }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700;800&display=swap" rel="stylesheet"/>
      <div style={{
        backgroundColor: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.08)",
        padding: "2rem",
        borderRadius: "12px",
        width: "100%",
        maxWidth: "400px"
      }}>
        <div style={{ marginBottom: "0.5rem", textAlign: "center" }}>
          <div style={{ fontSize: "11px", fontWeight: 700, letterSpacing: "2px", color: "#f59e0b", textTransform: "uppercase", marginBottom: "8px" }}>
            Wilder AI
          </div>
          <h1 style={{
            fontSize: "24px",
            fontWeight: 800,
            color: "#f8fafc",
            margin: 0
          }}>
            Tour Router
          </h1>
        </div>
        <div style={{
          fontSize: "13px",
          color: "#94a3b8",
          textAlign: "center",
          marginBottom: "2rem"
        }}>
          Enter password to continue
        </div>
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: "1rem" }}>
            <label style={{
              display: "block",
              marginBottom: "0.5rem",
              fontSize: "13px",
              fontWeight: 600,
              color: "#cbd5e1"
            }}>
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={{
                width: "100%",
                padding: "12px 16px",
                border: "1px solid rgba(255,255,255,0.12)",
                borderRadius: "8px",
                fontSize: "14px",
                backgroundColor: "rgba(255,255,255,0.05)",
                color: "#f8fafc",
                outline: "none",
                fontFamily: "inherit",
                boxSizing: "border-box"
              }}
              autoFocus
            />
          </div>
          {error && (
            <div style={{
              padding: "12px 16px",
              background: "rgba(239,68,68,0.1)",
              border: "1px solid rgba(239,68,68,0.3)",
              borderRadius: "8px",
              color: "#fca5a5",
              fontSize: "13px",
              marginBottom: "1rem"
            }}>
              {error}
            </div>
          )}
          <button
            type="submit"
            style={{
              width: "100%",
              padding: "12px",
              backgroundColor: "#f59e0b",
              color: "#0a0a0f",
              border: "none",
              borderRadius: "8px",
              fontSize: "14px",
              fontWeight: 700,
              cursor: "pointer",
              fontFamily: "inherit"
            }}
          >
            Login
          </button>
        </form>
      </div>
    </div>
  );
}
