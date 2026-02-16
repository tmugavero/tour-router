import { NextResponse } from "next/server";

export async function POST(request) {
  const { password } = await request.json();

  if (password === "wilder") {
    const response = NextResponse.json({ success: true });

    // Set a simple auth cookie
    response.cookies.set("auth", "authenticated", {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "strict",
      maxAge: 60 * 60 * 24 // 24 hours
    });

    return response;
  }

  return NextResponse.json(
    { error: "Invalid password" },
    { status: 401 }
  );
}
