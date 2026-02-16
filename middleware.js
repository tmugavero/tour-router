import { NextResponse } from "next/server";

export function middleware(request) {
  const authCookie = request.cookies.get("auth");
  const isLoginPage = request.nextUrl.pathname === "/login";
  const isApiRoute = request.nextUrl.pathname.startsWith("/api");

  // Allow API routes to pass through
  if (isApiRoute) {
    return NextResponse.next();
  }

  // If user is authenticated and trying to access login, redirect to home
  if (authCookie && isLoginPage) {
    return NextResponse.redirect(new URL("/", request.url));
  }

  // If user is not authenticated and not on login page, redirect to login
  if (!authCookie && !isLoginPage) {
    return NextResponse.redirect(new URL("/login", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
