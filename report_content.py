"""
Take-home swing report: content layer.

Architecture (three separate layers, so each does what it is best at):

    1. MEASUREMENT  -> SwingScorer in process_swing.py           (deterministic numbers)
    2. KNOWLEDGE    -> FIX_LIBRARY below                          (vetted golf coaching)
    3. LANGUAGE     -> polish_report() calls Claude               (personalises the prose)

The golf advice lives ENTIRELY in FIX_LIBRARY - fixed, human-written, correct.
Claude never invents a diagnosis or a drill; it only turns the assembled facts into
warm, personalised, booth-friendly prose. That keeps the two scariest failure modes
(hallucinated faults, hallucinated cures) off the table.

`render_template_report()` produces a complete, correct report with NO model call
(instant, safe, deterministic). `polish_report()` is the optional LLM pass on top; it
falls back to the template if Claude is unavailable, so the booth never shows an error.
"""
import json

# Only surface a fault worth practising. Small deviations (e.g. a pro's 168-deg
# address arm) are inside normal tolerance - coaching them reads as nagging.
COACH_THRESHOLD = 8.0     # strokes; a fault must cost at least this much to be coached
STRENGTH_THRESHOLD = 3.0  # strokes; a check this clean or cleaner is called a strength
MAX_FIXES = 3             # never dump more than this on someone at a booth


# ---------------------------------------------------------------------------
# KNOWLEDGE LAYER: one vetted entry per (phase, check). This is the golf brain.
#   plain -> what we saw, in plain words   why -> why it costs them
#   do    -> the corrective DIRECTION       drill -> a concrete thing to practise
#   feel  -> a one-line swing thought        title -> short headline
# ---------------------------------------------------------------------------
FIX_LIBRARY = {
    "address:correct_arm_angle": {
        "title": "Straighten the lead arm at setup",
        "plain": "Your lead arm was slightly bent when you set up to the ball.",
        "why": "A straight (but relaxed) lead arm fixes your swing radius, so the club "
               "returns to the same spot every time. Bending it at setup makes the low "
               "point wander, and that is fat or thin contact.",
        "do": "Let your lead arm hang long and relaxed from the shoulder - straight, not "
              "rigidly locked. Feel one straight line from your lead shoulder to the clubhead.",
        "drill": "Set up face-on to a mirror and check that your lead arm and the club make "
                 "a single straight line. Hold it for three breaths, then swing.",
        "feel": "Long lead arm, soft hands.",
        "action": "Make 5 slow setups in front of a mirror, holding a straight lead arm for 3 seconds each.",
    },
    "address:correct_midpoint": {
        "title": "Set your hands ahead of the ball",
        "plain": "Your hands were set behind the centre of your stance at address.",
        "why": "Hands slightly ahead of the ball preset a touch of forward shaft lean, which "
               "lets you hit the ball first and the turf second. Hands behind it promote scooping.",
        "do": "Position your hands so they sit just ahead of the ball, roughly over your lead "
              "thigh, with the shaft leaning a hair toward the target.",
        "drill": "Lay a club on the ground pointing at your lead hip pocket, and set up so your "
                 "hands match that line.",
        "feel": "Hands over the lead thigh.",
        "action": "Lay a club on the ground pointing at your lead hip, then set your hands to it for 10 reps.",
    },
    "top:correct_pelvis": {
        "title": "Finish your backswing turn",
        "plain": "At the top, your turn stopped a little short of a full coil.",
        "why": "A full turn - lead ankle, hip and trail shoulder stacking toward a straight "
               "line - stores power and widens your arc. Stopping short leaves speed on the table.",
        "do": "Turn your shoulders until your back faces the target and your lead shoulder "
              "rotates under your chin. Let your hips turn with them rather than sliding.",
        "drill": "Cross your arms on your chest and rotate back until your lead shoulder sits "
                 "over your trail foot. Feel the stretch across your back - five slow reps, then swing.",
        "feel": "Turn your back to the target.",
        "action": "Do 10 cross-arm chest turns until your lead shoulder is over your back foot, then hit 5 easy shots.",
    },
    "top:correct_head": {
        "title": "Keep your head steady going back",
        "plain": "Your head slid away from the target during the backswing.",
        "why": "If your head sways back, you have to time a slide forward to find the ball again "
               "- inconsistent contact. A steady head lets you rotate around a fixed centre.",
        "do": "Keep your head centred as you turn. Rotate your shoulders under a still head "
              "instead of swaying your whole upper body away from the target.",
        "drill": "Set up with your trail shoulder a few inches from a wall. Make backswings that "
                 "turn without your head drifting into it - turn, don't sway.",
        "feel": "Turn in a barrel.",
        "action": "Make 10 backswings with your trail shoulder near a wall, turning without touching it.",
    },
    "contact:correct_knee_angle": {
        "title": "Post up on your lead leg at impact",
        "plain": "Your lead leg stayed bent through impact.",
        "why": "Straightening your lead leg at impact clears room for your hips to rotate and "
               "adds speed. A soft lead knee usually means you are hanging back or sliding.",
        "do": "Coming into the ball, brace and straighten your lead leg - feel your weight stack "
              "onto a firm lead side while your hips clear out of the way.",
        "drill": "Slow-motion swings pausing at impact: check the lead leg is straight and your "
                 "belt buckle points ahead of the ball.",
        "feel": "Post up and clear.",
        "action": "Do 10 slow swings pausing at impact, checking your lead leg is straight each time.",
    },
    "contact:correct_arm_angle": {
        "title": "Extend through the ball (no chicken wing)",
        "plain": "Your lead arm folded (a 'chicken wing') through impact.",
        "why": "A bent lead elbow at impact collapses your radius and leaks both power and "
               "face control right where it counts. Both arms want to extend through the ball.",
        "do": "Feel both arms reach out toward the target through and just past impact, with your "
              "hands staying ahead of the clubhead. Extend, don't fold.",
        "drill": "Slow rehearsals that stop just after impact - check both arms are long and the "
                 "club points down the target line.",
        "feel": "Reach for the target.",
        "action": "Make 10 slow swings that stop just past impact with both arms fully extended down the line.",
    },
    "contact:correct_shoulder_ankle": {
        "title": "Rotate instead of sliding",
        "plain": "Your upper body slid past your front foot at impact.",
        "why": "When your lead shoulder drifts past your front foot you are sliding toward the "
               "target instead of turning - a big miss on both strike and direction.",
        "do": "Rotate around your lead side rather than sliding into it. Keep your lead shoulder "
              "stacked over your lead hip and let your hips turn open.",
        "drill": "Stand an alignment stick just outside your lead hip and make swings that rotate "
                 "without bumping it - rotate, don't lunge.",
        "feel": "Rotate, don't slide.",
        "action": "Stand an alignment stick just outside your lead hip and make 10 swings rotating without hitting it.",
    },
    "contact:correct_head": {
        "title": "Stay behind the ball through impact",
        "plain": "Your head drifted toward the target through impact.",
        "why": "Keeping your head behind the ball lets you deliver speed and a slightly upward "
               "strike. Letting it get ahead of the ball leads to thin, weak contact.",
        "do": "Stay behind the ball through the hit - feel your nose stay behind the ball position "
              "until well after contact.",
        "drill": "Pick a spot on the ground just behind the ball and keep your eyes on it until "
                 "the ball is gone.",
        "feel": "Head back, watch the spot.",
        "action": "Hit 10 balls keeping your eyes on a spot just behind the ball until it is gone.",
    },
}


def _measured_phrase(check, kind, scorer):
    """A human, number-carrying description of what was measured (the 'wow' specificity)."""
    if kind == "angle":
        return f"measured {check['actual']:.0f} degrees, where a perfectly straight line is 180"
    if kind == "head":
        pct = check["actual"] / scorer.body_scale * 100
        return f"your head moved about {pct:.0f}% of your body height from where it started"
    if kind == "pos_ge":
        return "you ended up on the wrong side of the line we look for here"
    return ""


def build_report_context(scorer, player="Player"):
    """
    Turn a SwingScorer into the structured facts the report is built from.
    Returns a plain dict (also what we hand Claude) - no prose yet.
    Leaderboard rank is deliberately NOT included: it lives on the live board,
    not on a personal take-home (it shifts per player and reads as discouraging).
    """
    result = scorer.score()
    kinds = {(c["phase"], c["name"]): c["kind"] for c in scorer.CHECKS}

    faults, strengths = [], []
    for c in result["checks"]:
        key = f"{c['phase']}:{c['name']}"
        lib = FIX_LIBRARY.get(key, {})
        if c["penalty"] >= COACH_THRESHOLD and lib:
            faults.append({
                "phase": c["phase"],
                "penalty": c["penalty"],
                "measured": _measured_phrase(c, kinds[(c["phase"], c["name"])], scorer),
                **lib,
            })
        elif c["penalty"] <= STRENGTH_THRESHOLD:
            strengths.append(c["label"])

    faults.sort(key=lambda f: f["penalty"], reverse=True)
    faults = faults[:MAX_FIXES]

    return {
        "player": player,
        "score": result["total"],
        "phase_scores": result["phase"],
        "strengths": strengths[:3],
        "fixes": faults,
    }


def render_template_report(ctx):
    """
    Deterministic, model-free take-home report (Markdown). Always correct, instant, safe.
    This is the baseline; polish_report() is the optional LLM pass on top of it.
    """
    L = []
    L.append(f"# {ctx['player']} - Swing Score: {ctx['score']:.1f}")
    L.append("*Lower is better (like golf). 0 would be a physically perfect swing - even tour "
             "pros don't get there, so this is your distance from flawless.*")
    L.append("")
    L.append("_[ address | top-of-backswing | impact frames go here ]_")
    L.append("")

    if ctx["strengths"]:
        L.append("## What you nailed")
        for s in ctx["strengths"]:
            L.append(f"- {s} \U0001F44D")
        L.append("")

    if ctx["fixes"]:
        L.append("## Your top fixes")
        for i, f in enumerate(ctx["fixes"], 1):
            tag = " (biggest point-saver)" if i == 1 else ""
            L.append(f"### {i}. {f['title']}{tag}")
            L.append(f"**What we saw:** {f['plain']} ({f['measured']}.)")
            L.append(f"**Why it matters:** {f['why']}")
            L.append(f"**What to do:** {f['do']}")
            L.append(f"**Drill:** {f['drill']}")
            L.append(f"**Swing thought:** *{f['feel']}*")
            L.append("")
    else:
        L.append("## Nice swing!")
        L.append("Nothing major to fix from what we measured - keep grooving it.")
        L.append("")

    if ctx["fixes"]:
        L.append("## Your practice plan")
        L.append("Do these in order next time you're at the range:")
        for i, f in enumerate(ctx["fixes"], 1):
            L.append(f"{i}. **{f['title']}** - {f['action']} *({f['feel']})*")
        L.append("")

    L.append("---")
    L.append("_Automated estimate from a single camera - a fun snapshot, not a substitute for "
             "a lesson from a coaching pro._")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# LANGUAGE LAYER: optional Claude polish. Personalise ONLY - never re-diagnose.
# ---------------------------------------------------------------------------
POLISH_SYSTEM = (
    "You are a seasoned PGA teaching professional giving a friendly end-of-lesson recap to "
    "someone who just took a swing at an AI expo booth. You are given (a) structured facts "
    "about their swing and (b) a draft report. Rewrite it in your own voice - the way a good "
    "instructor actually talks on the lesson tee.\n\n"
    "VOICE - sound like a real teaching pro:\n"
    "- Write in the second person and the imperative, addressed straight to the golfer. Do NOT use "
    "the first person ('I', 'me', 'my', 'I'd', 'let me') - this is coaching directed at them, not a "
    "pro narrating themselves.\n"
    "- Open by reading their swing back to them ('Here's what your swing showed...', 'Watching this "
    "one back...'), speaking directly to them as 'you'.\n"
    "- Lead with what is already working before any fix - genuine, specific praise, not filler.\n"
    "- Give them ONE priority to groove first ('if you work on one thing before you play again, make "
    "it this'); frame the rest as supporting it.\n"
    "- Coach in feels and simple images, not mechanics jargon. Many readers have never golfed, so when "
    "a golf term is unavoidable, gloss it in plain words the way you would for a beginner.\n"
    "- Be warm, direct, and confident - never clinical or discouraging. A fault is the next thing to "
    "work on, not a failing.\n"
    "- End with a clear, numbered practice plan: the concrete, countable actions provided for each fix "
    "(the 'action' items), in priority order, so they leave knowing exactly what to do at the range. "
    "Then one encouraging line sending them off.\n\n"
    "HARD RULES (never break these):\n"
    "- Use ONLY the faults, drills, and numbers provided. Never invent a fault, a drill, a cause, "
    "or a statistic. If a fact is not in the input, do not state it.\n"
    "- Keep every measured number (e.g. '166 degrees') exactly - that specificity is what makes "
    "your read credible.\n"
    "- Keep it to roughly one page, and keep the closing disclaimer. Output GitHub-flavoured Markdown."
)


def polish_report(ctx, draft_md, model="claude-opus-4-8"):
    """
    Optional Claude pass that personalises `draft_md`. Falls back to the draft on ANY
    problem (no SDK, no credentials, API error) so the booth never shows a failure.
    Returns (text, used_llm: bool).
    """
    try:
        import anthropic
        client = anthropic.Anthropic()
        user = (
            "Structured facts (JSON):\n"
            + json.dumps(ctx, indent=2)
            + "\n\nDraft report to rewrite:\n\n"
            + draft_md
        )
        resp = client.messages.create(
            model=model,
            max_tokens=1500,
            output_config={"effort": "low"},  # short copy task; keep it fast/cheap for a booth queue
            system=POLISH_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "").strip()
        return (text or draft_md), bool(text)
    except Exception as e:
        return draft_md, False
