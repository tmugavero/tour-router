export async function POST(req) {
  const { artist } = await req.json();

  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": process.env.ANTHROPIC_API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      messages: [{
        role: "user",
        content: `You are a live music industry analyst. For the artist "${artist}", estimate their current touring profile. Respond ONLY with a JSON object, no markdown, no backticks:\n\n{"artist_name":"exact artist name","growth_phase":"emerging|club|club_to_theater|theater|arena","estimated_capacity":<typical headline venue capacity integer>,"estimated_guarantee":<typical guarantee per show USD integer>,"estimated_fill_rate":<0.0 to 1.0>,"genre":"primary genre","comparable_artists":["3 similar-tier touring artists"],"summary":"1-2 sentence explanation of touring status and these numbers"}\n\nBe precise about real venue sizes and guarantee ranges for their actual current stature.`
      }]
    })
  });

  const data = await res.json();
  return Response.json(data);
}
