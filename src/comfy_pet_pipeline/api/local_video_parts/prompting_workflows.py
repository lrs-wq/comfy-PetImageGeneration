"""Prompt construction, retry profiles, and Comfy workflow builders."""

from __future__ import annotations

# The extracted functions resolve the facade dependencies after runtime sync.
# ruff: noqa: F821

from .runtime import sync_local_video_runtime as _sync_local_video_runtime


def merge_image_profile_loras(
    profile: dict[str, Any] | None,
    requested: list[dict[str, Any]],
    *,
    requested_mode: str,
    resolve_models: bool = True,
) -> tuple[list[dict[str, Any]], str, str]:
    """Apply baseline LoRAs, allowing uploaded LoRAs to replace or stack them."""
    _sync_local_video_runtime(globals(), {"merge_image_profile_loras"})
    if profile is None:
        return [dict(item) for item in requested], requested_mode, "request"
    profile_loras = normalize_user_lora_specs(
        profile["user_loras"],
        resolve_models=resolve_models,
    )
    if not requested:
        return profile_loras, str(profile["user_lora_mode"]), "image_profile"
    if requested_mode == "replace":
        return [dict(item) for item in requested], requested_mode, "user_override"
    combined = [*profile_loras, *requested]
    normalized = normalize_user_lora_specs(
        combined,
        resolve_models=resolve_models,
    )
    return normalized, requested_mode, "image_profile_plus_user"


def cap_user_lora_specs(
    specs: list[dict[str, Any]],
    *,
    total_cap: float,
    per_lora_cap: float,
) -> tuple[list[dict[str, Any]], float]:
    """Scale all LoRAs proportionally so stage-specific caps cannot lock the pose."""
    _sync_local_video_runtime(globals(), {"cap_user_lora_specs"})
    total_strength = sum(float(item["strength"]) for item in specs)
    maximum_strength = max((float(item["strength"]) for item in specs), default=0.0)
    if total_strength <= 0.0 or maximum_strength <= 0.0:
        return [dict(item) for item in specs], 1.0
    scale = min(
        1.0,
        total_cap / total_strength if total_cap > 0.0 else 0.0,
        per_lora_cap / maximum_strength if per_lora_cap > 0.0 else 0.0,
    )
    scaled = [
        {**item, "strength": round(float(item["strength"]) * scale, 6)}
        for item in specs
    ]
    return scaled, scale


def reduced_user_lora_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve blend ratios while capping the combined recovery influence."""
    _sync_local_video_runtime(globals(), {"reduced_user_lora_specs"})
    reduced, _ = cap_user_lora_specs(
        specs,
        total_cap=QWEN_RECOVERY_USER_LORA_STRENGTH,
        per_lora_cap=2.0,
    )
    return reduced


def action_key(pose: str) -> str:
    _sync_local_video_runtime(globals(), {"action_key"})
    return pose.split("|", 1)[0].strip().lower()


def subject_replacement_pose_prompt(action: str, *, species: str = "cat") -> str:
    _sync_local_video_runtime(globals(), {"subject_replacement_pose_prompt"})
    if action == "eating":
        if species == "dog":
            return (
                "eating | A complete full-body dog in a natural low-head eating "
                "pose at a three-quarter front angle. Exactly one shallow food bowl "
                "contains a small, low, level layer of ordinary brown dry kibble "
                "fully below the rim. The food is never a heap, mound, cone, pyramid, "
                "fused mass, oversized pile, or overflow. The lower muzzle enters and "
                "overlaps the low food layer while all four anatomically attached legs "
                "remain stable and the complete tail stays visible."
            )
        return f"eating | {MULTIPOSE_PRESET_BY_ACTION['eating']['prompt_en']}"
    if action == "grooming":
        return SUBJECT_REPLACEMENT_GROOMING_POSE_PROMPT
    if action == "standing" and species == "dog":
        return (
            "standing | A complete full-body dog in the stable idle pose shown by "
            "the supplied action geometry. The geometry, rather than the action "
            "label, controls whether the torso is seated or upright. Keep all "
            "visible legs anatomically attached, both ears and the complete tail "
            "readable. Show a relaxed natural panting expression: the mouth is "
            "visibly open with a dark mouth cavity, separated lower jaw and one "
            "anatomically attached pink tongue at a neutral mid-extension. Never "
            "place a tongue through closed lips. No prop, clothing, collar, harness, "
            "floor, or extra subject."
        )
    if action == "drinking" and species == "dog":
        return (
            "drinking | A complete full-body dog in the low-head drinking pose "
            "shown by the supplied action geometry at a three-quarter front angle. "
            "Exactly one shallow water bowl stays immediately in front of the paws. "
            "The muzzle is lowered to the water and one tiny anatomically attached "
            "pink tongue tip touches the water surface. Keep all four legs attached, "
            "both ears and the complete tail readable. No food, collar, clothing, "
            "floor, shadow, or extra subject."
        )
    if action == "running" and species == "dog":
        return (
            "running | A complete full-body dog at a three-quarter side angle in "
            "the exact quadruped running contact phase shown by the supplied action "
            "geometry. Keep exactly four distinct anatomically attached legs and "
            "complete paws, a coherent torso, both ears, and the tail continuously "
            "visible from base to tip. No prop, floor, shadow, clothing, collar, "
            "harness, or extra subject."
        )
    if action == "sleeping" and species == "dog":
        return (
            "sleeping | A complete full-body dog in the compact curled sleeping pose "
            "shown by the supplied action geometry. The torso rests low, the head "
            "rests naturally beside or on the front paws, the eyes are peacefully "
            "closed, the legs remain anatomically attached, and the complete tail "
            "curls continuously beside the body. No bed, cushion, floor, shadow, "
            "collar, clothing, or extra subject."
        )
    if action == "toileting" and species == "dog":
        return (
            "toileting | A complete full-body dog in the stable seated toilet pose "
            "shown by the supplied action geometry, centered on exactly one clean "
            "rectangular pet toilet tray with a visible grid insert. Keep all four "
            "legs anatomically attached, both ears, the complete tail, and the tray "
            "fully visible. The clean tray is the only prop. Show no urine, feces, "
            "mess, floor, shadow, collar, clothing, or extra subject."
        )
    raise ValueError(f"Unsupported subject-replacement action: {action}")


def action_uses_bowl(pose: str) -> bool:
    _sync_local_video_runtime(globals(), {"action_uses_bowl"})
    return action_key(pose) in {"eating", "drinking"}


def pose_seed_offset(pose: str, index: int) -> int:
    _sync_local_video_runtime(globals(), {"pose_seed_offset"})
    action = action_key(pose)
    canonical_action = "lying down" if action == "lying" else action
    return MULTIPOSE_SEED_OFFSETS.get(canonical_action, max(1, index))


def qwen_first_frame_spec(pose: str) -> str:
    _sync_local_video_runtime(globals(), {"qwen_first_frame_spec"})
    action = action_key(pose)
    detail = pose.split("|", 1)[1].strip() if "|" in pose else ""
    detail = re.split(r";\s*video\s*:", detail, maxsplit=1, flags=re.IGNORECASE)[
        0
    ].strip()
    detail = re.sub(r"^first\s+frame\s*:\s*", "", detail, flags=re.IGNORECASE)
    if action == "running":
        safety_spec = MULTIPOSE_PRESET_BY_ACTION["running"]["prompt_en"]
        if detail.casefold() == safety_spec.casefold():
            return safety_spec
        return (
            f"{safety_spec}; requested first-frame detail: {detail}"
            if detail
            else safety_spec
        )
    if not detail:
        detail = pose.strip()
    return detail or pose.strip()


def action_first_frame_rules(pose: str, *, species: str = "cat") -> str:
    _sync_local_video_runtime(globals(), {"action_first_frame_rules"})
    action = action_key(pose)
    if species == "dog" and action == "eating":
        return (
            "Include exactly one small shallow matte food bowl containing only a "
            "small, low, level layer of ordinary brown dry kibble fully below its "
            "continuous rim. Keep the food surface shallow and planar: no heap, mound, "
            "cone, pyramid, oversized pellets, fused mass, overfilled bowl, or food "
            "outside the bowl. Keep all four dog legs anatomically attached and stable. "
            "Lower the head until the dog's lower muzzle physically enters and overlaps "
            "the low kibble layer and near rim; a gap between mouth and food is invalid. "
            "Keep both ears, the upper face, body and complete tail readable. The bowl "
            "is the only prop."
        )
    if species == "dog" and action == "standing":
        return (
            "No prop. Match Image 2's stable dog idle geometry exactly, including "
            "its seated-or-upright torso state, leg placement, head angle, camera, "
            "subject scale and tail layout. Keep every visible paw grounded and "
            "every leg attached. "
            "Show one complete natural dog with both ears and the tail from base to "
            "tip; no pose redesign, extra limb, collar, harness, clothing or object. "
            "This action explicitly requires relaxed panting even if Image 1 is a "
            "closed-mouth still: keep the lips and lower jaw visibly separated, show "
            "a readable dark mouth cavity, and attach one natural pink tongue inside "
            "the mouth at a neutral mid-extension. Never paste a tongue onto a closed "
            "muzzle or detach it from the mouth cavity. "
            "The dog exists in a digital green void with no support surface: every "
            "paw and the tail must end cleanly against pure green, with an "
            "uninterrupted green band below them. Never draw a floor, contact shadow, "
            "mirrored continuation, glossy patch or reflection beneath the dog."
        )
    if species == "dog" and action == "drinking":
        return (
            "Include exactly one small shallow matte water bowl directly in front "
            "of the dog. Match Image 2's low-head drinking geometry, bowl placement, "
            "camera, scale and framing. Lower the muzzle naturally to the still water "
            "and show exactly one tiny attached pink tongue tip touching the water; "
            "no air gap, long tongue, splash, drool, food, extra bowl or extra prop. "
            "Keep all four legs attached and the complete tail readable."
        )
    if species == "dog" and action == "running":
        return (
            "No prop. Match Image 2's quadruped running contact phase, camera, scale "
            "and framing. Show exactly four distinct anatomically attached legs, "
            "each ending in one complete paw; keep the torso coherent and the full "
            "tail connected from base to tip. Do not fuse or hide the far-side legs, "
            "turn the dog into a two-legged leap, crop a paw, add a floor, or add a "
            "second subject."
        )
    if species == "dog" and action == "sleeping":
        return (
            "No prop. Match Image 2's compact curled sleeping geometry exactly. Keep "
            "the head resting naturally by the front paws, eyes peacefully closed, "
            "all legs anatomically attached and tucked with plausible occlusion, and "
            "the tail continuous from base to tip beside the body. Do not add a bed, "
            "cushion, blanket, floor, shadow, open alert eyes, or extra subject."
        )
    if species == "dog" and action == "toileting":
        return (
            "Match Image 2's stable seated toilet geometry and keep exactly one clean "
            "rectangular pet toilet tray with its grid insert fully visible beneath "
            "the dog. The tray is the only prop. Keep all four legs attached, both "
            "ears and the complete tail readable. Show no urine, feces, mess, extra "
            "tray, litter box, floor continuation, shadow, clothing, or extra subject."
        )
    if action == "running":
        return (
            "No prop. This still image is a calm static pre-run standing pose for later animation, not an illustrated "
            "running instant, jump, leap, gallop suspension, or flying pose. Use a three-quarter side view with the "
            "camera pulled back. The torso is upright and visibly supported above the legs. Plant and visibly flatten "
            "all four complete paws on the same horizontal baseline with no gap below them. Show exactly four long "
            "legs, each continuously attached to the torso, spatially separated, staggered front-to-back, and ending "
            "in one complete paw. Offset the far-side paws horizontally so they remain readable rather than overlapping "
            "the near-side paws. Each leg must be at least one quarter of the torso height. "
            "Keep both hind legs substantial and fully formed; do not hide, shorten, fuse, overlap, fragment, or "
            "detach any leg or paw. The head turns only slightly toward camera so both eyes and identity markings remain "
            "readable. At least three paws must make unambiguous ground contact in the final silhouette; any airborne "
            "body pose invalidates the image."
        )
    if action == "eating":
        return (
            "Include exactly one small shallow matte food bowl, touching or slightly "
            "overlapping the front paws so it remains one compact cutout. Fill the bowl "
            "with clearly visible brown kibble and lower the head until the lower muzzle "
            "and mouth physically enter the food. Require 6 to 10 output pixels of "
            "visible muzzle-to-kibble overlap: the nose or lower muzzle must be partly "
            "occluded by kibble and the near rim. Any green, white, or background gap "
            "between mouth and food is invalid. This is active eating, never looking at, "
            "sniffing, or hovering above the bowl. The bowl is the only prop. Do not let the "
            "bowl hide or replace any limb. Both complete front paws remain visibly "
            "planted; the cat never grasps, lifts, carries, or holds food. Keep the ears, "
            "upper head markings, paws, body, and full tail readable, but do not raise the "
            "head to make the complete eyes, nose, or face readable."
        )
    if action == "drinking":
        return (
            "Include exactly one small shallow matte water bowl immediately beside the front paws, with a narrow visible "
            "green gap so the bowl remains a separate foreground component. The bowl is the only prop. Do not let it "
            "touch, overlap, hide, or replace a paw. Show a recognizable active-drinking keyframe: the "
            "muzzle is lowered naturally and exactly one tiny solid-pink triangular cat tongue tip emerges from the "
            "lower lip and visibly intersects the still water with no air gap. Its visible length must not exceed one "
            "twentieth of the head width. The tongue must look anatomical and remain visibly pink, never transparent, "
            "blue, elongated, or detached. Keep both eyes, the "
            "face silhouette, paws, and body readable."
        )
    if action == "grooming":
        return (
            "No prop. Use a stable seated grooming pose. Exactly one foreleg rises "
            "continuously from its visible shoulder to one paw touching the mouth, and "
            "the other foreleg ends in exactly one planted front paw. The raised leg "
            "must not leave a duplicate paw on the ground. Both hind legs remain "
            "anatomically attached and may be naturally tucked at the sides; a seated hind paw "
            "may be partly occluded, but no leg or paw may be duplicated or detached. "
            "Show one tiny pink tongue touching the raised paw. No food, bowl, detached "
            "paw, ghost limb, mouth object, or extra subject."
        )
    if action in {"lying", "lying down"}:
        return "No prop. Use a relaxed low full-body pose; do not crop the paws, belly, haunches, or tail."
    return "Do not add a prop unless the action specification explicitly requires one."


def action_video_rules(pose: str) -> str:
    _sync_local_video_runtime(globals(), {"action_video_rules"})
    action = action_key(pose)
    canonical_action = "lying down" if action == "lying" else action
    preset = MULTIPOSE_PRESET_BY_ACTION.get(canonical_action)
    if preset:
        return preset["video_prompt_en"]
    return "Use only small action-appropriate body motion while the torso root, head design, and camera remain stable."


def build_cartoon_reference_prompt(
    base_prompt: str = "",
    *,
    palette_lock: str = "",
    fur_lock: str = "",
    species: str = "cat",
) -> str:
    _sync_local_video_runtime(globals(), {"build_cartoon_reference_prompt"})
    creative_direction = (
        base_prompt.strip() or "identity-faithful restrained stylized 3D digital double"
    )
    palette_section = f"\n\n{palette_lock.strip()}" if palette_lock.strip() else ""
    if species == "dog":
        fur_section = f"\n\n{fur_lock.strip()}" if fur_lock.strip() else ""
        return f"""Use case: identity-preserving stylized 3D dog character design.
BACKGROUND RULE — non-negotiable: every pixel outside the dog must be one perfectly flat uniform pure #00FF00 green. It is a blank solid-color field, never a poster or graphic: no decorative shape, icon, symbol, letter, number, outline, pattern, floor, shadow, gradient, scenery, text, logo, watermark, border, or halo. No green shape, reflection, marking, or outline may cross onto or appear inside the dog's silhouette.

The uploaded identity image, or the role-labeled uploaded identity image set when multiple views are supplied, is the only identity and appearance authority. Create a canonical identity-faithful restrained semi-realistic stylized 3D digital double of this exact same dog. Never copy identity, breed, proportions, face, coat, color, markings, ears, eyes, muzzle, tail, material, or lighting from an action profile or white model.

User creative direction: {creative_direction}
Apply it only to restrained rendering quality and expression. It cannot override identity, anatomy, full-body framing, natural coat, clothing, or background rules.

Identity lock: preserve the uploaded dog's exact apparent age, breed traits, skull width, forehead slope, cheek width, head-to-body ratio, muzzle length, muzzle width and taper, eye diameter and spacing, ear size and carriage, nose size and color, mouth posture, coat base colors, localized markings, natural coat length and volume, body proportions, leg length, paws, and complete tail. Preserve natural asymmetry and never invent markings. The front view is authoritative for the face and expression; side views are authoritative for muzzle projection, body length, tail and side-specific markings.{palette_section}{fur_section}

Target style: a polished restrained semi-realistic stylized 3D digital double under neutral-white high-key character lighting. Identity outranks stylization. Preserve an adult as an adult and a puppy as a puppy; never shorten the muzzle, round the forehead, enlarge the eyes or head, add a panting smile, or otherwise make the dog younger, chibi, toy-like, feline, or generic. Preserve the uploaded front view's closed- or open-mouth state unless an action explicitly requires mouth motion. The dog is completely natural and unclothed: no collar, harness, glasses, bow, shirt, costume, shoes, jewelry, or accessory.

Composition lock: one complete dog in a neutral stable upright four-paw pose at a slight three-quarter front angle. Show both ear tips, the full torso, exactly four anatomically attached legs and complete paws, and the tail continuously from base to tip. Keep at least 15% clear green margin on every edge and keep the dog within 68% of canvas width and height. Nothing may be cropped, fused, hidden behind the torso, or outside the canvas.

{CHROMA_BACKGROUND_PROMPT} No floor, floor line, shadow, reflection, text, logo, or watermark."""
    fur_section = ""
    if fur_lock.strip():
        fur_section = f"""

{fur_lock.strip()}

UPLOAD COAT SURFACE PRESERVATION — non-negotiable: the uploaded views, not the
action profile or model style, control natural coat length, volume, edge grooming,
and strand direction. Keep the cheeks, neck, chest, shoulders, back, flanks,
haunches, and tail consistent with the uploaded pet. Do not infer missing body fur
from a profile animal and do not turn a cropped view into a mane, ruff, plume,
radial brush strokes, separated filament bundles, a wispy halo, or a different
coat-length category."""
    return f"""Use case: identity-preserving stylized 3D character design.
BACKGROUND RULE — non-negotiable: every pixel outside the character must be one perfectly flat uniform pure #00FF00 green. No black, gray, white, studio, or environmental background; no floor and no shadow.

The uploaded identity image, or the role-labeled uploaded identity image set when multiple views are supplied, is the only identity reference. Create a canonical identity-faithful restrained stylized 3D digital double of this exact same cat. Keep the cat recognizable and change only the rendering medium, never the cat's identity, apparent age, or natural appearance.

User creative direction: {creative_direction}
Apply this direction only to rendering quality, expression, and lighting on the character. It must not override the identity, coat length or volume, full-body anatomy, composition, clothing, or background rules below.
Ignore every pose, action, camera angle, framing, crop, background, and subject-placement instruction found in the user creative direction. The canonical pose, composition, and background rules below are authoritative.

Identity lock: preserve the exact fur base colors, visible markings and their placement, head and face silhouette, cheek width, natural head-to-body ratio, eye size, eye color and spacing, ear size and shape, nose color, muzzle, breed traits, natural coat length and volume, body proportions, and tail from the uploaded cat. The selected action profile is never an appearance reference. Preserve natural asymmetry. Do not turn lighting, shadows, blur, or fur texture into new markings. Do not invent stripes or spots.{palette_section}{fur_section}

Target style: restrained semi-realistic stylized 3D rendering applied to the exact uploaded pet, with physically plausible materials, source-matched natural fur, color-faithful neutral high-key lighting on the character, clean catchlights, and a calm expression. Identity and apparent age outrank stylization: keep the source head size, cheek width, eye diameter, iris diameter, eye spacing, ear size, muzzle length, and body ratio. Preserve the uploaded pet's mature or juvenile appearance exactly; never make it younger or more kitten-like. Do not use chibi enlargement, oversized eyes, enlarged irises, oversized ears, rounded baby-face proportions, or a generic mascot face. Lighting must never tint or replace the source fur palette. The result should look like the same recognizable pet rendered as a polished 3D digital double rather than a redesigned mascot, real photograph, flat drawing, or plastic toy.

Grooming rule: {SLEEK_GROOMING_PROMPT}

Natural cat only: the character is completely unclothed and wears no glasses, collar, bow tie, shirt, overalls, suspenders, harness, hat, shoes, jewelry, costume, or other accessory. Show the natural fur across the neck, chest, torso, and legs. Do not import the example image's wardrobe or eyewear.

Composition lock: this is a pulled-back full-body character sheet, never a portrait or close-up. Show one complete cat from both ear tips through the torso to every paw and the full tail tip. Use a neutral compact standing pose at a slight three-quarter front angle. Show exactly four anatomically attached legs and four complete paws with visible separation; show the tail continuously from base to tip. Nothing may be hidden behind the torso, fused, shortened, cropped, fragmented, or outside the canvas. Center the complete silhouette with at least 15% clear green margin on every side and let the cat occupy no more than 68% of the canvas width and height. Both eyes are fully open, equally sized, fully visible and level, pupils centered, face neutral and symmetrical, mouth closed. The fur silhouette must meet green directly with no white border or halo.

{CHROMA_BACKGROUND_PROMPT} The pure green background rule overrides all aesthetic lighting language. Keep the cat at a useful readable scale. No floor, shadow, text, logo, or watermark."""


def build_grooming_foreleg_repair_prompt(
    pose: str,
    *,
    source_attempt: int,
    profile_fur_lock: str = "",
) -> str:
    """Complete a seeded grooming foreleg from identity and anatomy references."""
    _sync_local_video_runtime(globals(), {"build_grooming_foreleg_repair_prompt"})

    fur_lock = profile_fur_lock.strip() or (
        "Preserve Image 2's exact natural coat length, visible volume, strand "
        "direction, genuine tufts, and source-bounded silhouette."
    )
    return f"""CRITICAL GROOMING FORELEG COMPLETION - perform this local edit before every other detail:
Image 1 is the best identity-safe near-final candidate from attempt {source_attempt} and is the sampler canvas. Its seated body, head angle, face placement, tail, planted legs, scale, and full-body framing are authoritative. The bounded region between the viewer-left chest and mouth contains a deliberately band-limited identity-matched coat-color, mid-scale fur, and shading seed sampled from Image 2's planted foreleg, rotated so its paw end faces the mouth and its upper-leg end faces the chest. It supplies only approximate foreleg occupancy, the pet's color family, and genuine fur-material cues; it is not finished texture and none of its boundary or individual sampled strand pixels may be copied literally.
Image 2 is the approved Qwen full-body canonical and is the sole authority for identity and appearance: exact face, eyes, ears, nose, coat palette, localized markings, body proportions, natural fur surface, paws, tail, and restrained stylized 3D material.
Image 3 is a frozen neutral low-frequency anatomy map for this grooming pose. Copy only its shoulder-to-elbow-to-wrist-to-paw arrangement, head-to-paw contact, seated balance, and limb count. Its gray values are abstract geometry labels, never fur colors, markings, lighting, texture, material, clothing, or identity.

LOCAL COMPLETION CONTRACT: a generously expanded feathered full-denoise mask covers the seed plus ordinary surrounding muzzle, shoulder, and chest context. The mask is an editable workspace, not the foreleg silhouette: never fill its whole area with a limb or preserve its boundary. Leave every valid part of Image 1 outside that mask unchanged. Reconstruct the masked region as one continuous cat surface from Image 2: exactly one anatomically continuous furry raised foreleg, the natural shoulder/chest attachment around it, the small mouth-contact boundary, and pure green wherever no cat belongs. Use the seed only for approximate occupancy and identity palette; replace all seed edges, internal strokes, and sampled lighting with newly coherent fur generated from Image 2. The foreleg must emerge naturally from the viewer-left shoulder/chest, taper through a readable soft elbow and wrist, and end in one rounded paw touching the mouth. Match the planted paw's soft outline, short dense fur, and subtle toe structure. Join shoulder, wrist, paw, muzzle contact, and surrounding chest fur seamlessly; no straight or uniform-width artificial contour, cutout edge, internal gap, repeated radial strand fan, or unblended patch may remain.

LOCAL SURFACE PRIOR CONTRACT: the loaded low-strength Pet3D adapter is permitted only as a fine surface prior inside the sampler noise mask. Use it to break up smooth plastic or tube-like surfaces into compact irregular short fur, a soft paw outline, and restrained toe relief. It is never an identity, palette, marking, anatomy, lighting, or silhouette authority; Image 2 overrides it in every conflict. Do not lengthen the coat, inflate the limb, add brush-like strands, create radial fur fans, or alter any unmasked pixel.

IDENTITY LOCK: copy Image 2 only for every rendered pixel of the completed foreleg and for any small boundary refinement. Preserve Image 1's already valid face and composition. Use neutral-white character lighting. {fur_lock}

POSE CONTRACT: {qwen_first_frame_spec(pose)}
{action_first_frame_rules(pose)}
Exactly one foreleg is raised continuously from shoulder to mouth and exactly one other front paw is planted. Keep both attached hind legs and the complete tail. A tiny pink tongue may touch the raised paw. No duplicate ground paw, ghost limb, detached paw, food, bowl, prop, clothing, accessory, or extra subject.

OUTPUT CONTRACT: return exactly one clean static 512x512 first frame. Keep the complete cat within 72% of canvas width and height with at least 12% clear green margin on every edge. Every pixel outside the completed cat must be flat pure #00FF00. No floor, floor line, shadow, reflection, scenery, text, logo, or watermark. Return the completed grooming keyframe only."""


def build_canonical_grooming_foreleg_completion_prompt(
    pose: str,
    *,
    source_attempt: int,
    profile_fur_lock: str = "",
    soft_depth_anatomy: bool = False,
) -> str:
    """Locally add the Main_case raised foreleg to an identity-safe base."""
    _sync_local_video_runtime(globals(), {"build_canonical_grooming_foreleg_completion_prompt"})

    fur_lock = profile_fur_lock.strip() or (
        "Preserve Image 2's exact natural coat length, visible volume, strand "
        "direction, genuine tufts, and source-bounded silhouette."
    )
    anatomy_contract = (
        "Image 3 is the frozen normalized neutral-depth action anatomy. It has "
        "no identity or appearance role. Its connected low-frequency relief "
        "describes one living cat from head through torso and limbs. Read its "
        "viewer-left shoulder-to-elbow-to-wrist-to-paw path and paw-to-muzzle "
        "contact as continuous feline anatomy. Use Image 2 for every visible "
        "surface pixel inside the edited region."
        if soft_depth_anatomy
        else (
            "Image 3 is the frozen appearance-free single-gray action silhouette. "
            "It has no identity or appearance role. Read only its raised "
            "viewer-left shoulder-to-elbow-to-wrist-to-paw path, paw-to-muzzle "
            "contact, seated joint layout, and exterior green negative space. "
            "Use Image 2 for every visible surface pixel inside the edited region."
        )
    )
    return f"""CRITICAL MASKED GROOMING COMPLETION - perform only the requested local anatomy edit:
Image 1 is the best identity-palette-safe normalized candidate from attempt {source_attempt} and is the sampler canvas. Its face, eyes, ears, head shape, seated body, viewer-right planted front paw, hind legs, tail, scale, framing, pure-green canvas, colors, markings, coat surface, and rendering style are already correct. Preserve every pixel outside the sampler noise mask.
Image 2 is the approved Qwen full-body canonical and is the sole authority for identity and appearance inside the mask: exact face, eyes, ears, nose, coat palette, localized markings, body proportions, natural fur surface, paws, tail, and restrained stylized 3D material.
{anatomy_contract}

MASK CONTRACT: the feathered full-denoise mask is only an editable workspace around the viewer-left muzzle, shoulder, chest and missing foreleg. It is not a limb silhouette and must not be filled as one solid column. Outside the mask, copy Image 1 byte-for-byte in appearance. Inside the mask, use Image 2's identity and fur to replace the neutral two-paws-down anatomy with exactly one organic viewer-left foreleg visibly attached at the shoulder near (38% width, 67% height), bending through a readable soft elbow and wrist, and ending in one rounded furry paw centered near (37% width, 51% height) physically touching the viewer-left underside of the muzzle. Keep the viewer-right front paw planted. Remove any obsolete viewer-left ground paw only where the mask reaches it. Preserve natural green negative space around the curved limb; never preserve the mask boundary.

SURFACE AND IDENTITY LOCK: the completed foreleg must use Image 2's exact local coat palette, short/medium/long fur category, strand direction, soft paw outline and restrained toe relief. Build natural shoulder width, elbow taper, wrist taper and a rounded paw, with continuous fur direction across every joint. Join muzzle, paw, wrist, shoulder and chest seamlessly. The edited region contains only natural cat anatomy and green negative space, with no duplicate paw or new marking. Image 2 remains the sole appearance authority. Use neutral-white character lighting. {fur_lock}

POSE CONTRACT: {qwen_first_frame_spec(pose)}
{action_first_frame_rules(pose)}
Exactly one viewer-left foreleg is raised continuously from shoulder to mouth and exactly one viewer-right front paw is planted. Keep both attached hind legs and the complete tail. A tiny pink tongue may touch the raised paw. No duplicate ground paw, ghost limb, detached paw, food, bowl, prop, clothing, accessory, or extra subject.

OUTPUT CONTRACT: return exactly one clean static 512x512 first frame. Keep the complete cat within 72% of canvas width and height with at least 12% clear green margin on every edge. Every pixel outside the cat must be flat pure #00FF00. No floor, floor line, shadow, reflection, scenery, text, logo, or watermark. Return the completed grooming keyframe only."""


def build_canonical_grooming_full_pose_rescue_prompt(
    pose: str,
    *,
    profile_fur_lock: str = "",
) -> str:
    """Force the verified Main_case grooming skeleton before identity repaint."""
    _sync_local_video_runtime(globals(), {"build_canonical_grooming_full_pose_rescue_prompt"})

    fur_lock = profile_fur_lock.strip() or (
        "Preserve Image 2's exact natural coat length, visible volume, strand "
        "direction, genuine tufts, and source-bounded silhouette."
    )
    return f"""CRITICAL FULL-POSE GROOMING RESCUE - geometry first, identity exact:
Image 1 is the frozen appearance-free action geometry canvas and the sampler. It is authoritative only for the complete outer seated silhouette, viewer-left shoulder-to-elbow-to-wrist-to-paw path, paw-to-muzzle contact, viewer-right planted front paw, both attached hind legs, tail, framing, scale, and pure-green negative space. Every subject pixel is the same synthetic RGB(144,144,144); there is no internal luma pattern, sampled Main_case color, coat patch, marking, texture, material, clothing, identity, or hard object. Its one thin pure-green open curved channel is background negative space between the raised viewer-left foreleg and chest. It remains open at the paw and shoulder, exists only to make the organic bent limb readable, and must never be rendered as a green line, seam, hole, marking, microphone, hanging object, or replacement coat design.
Image 2 is the approved Qwen full-body canonical and is the sole identity and appearance authority. Render every cat pixel from Image 2: exact face, green eyes, ears, nose, coat palette, localized markings, body proportions, natural fur category and surface, paws, tail, restrained stylized 3D material, and neutral-white lighting. Never average Image 1's gray into Image 2 and never invent a replacement identity.

FULL-POSE CONTRACT: keep Image 1's verified action geometry as one organic cat, but completely regenerate its appearance from Image 2. Exactly one viewer-left foreleg must emerge continuously from the viewer-left shoulder near (38% width, 67% height), bend through a readable soft elbow and wrist, and end in one rounded furry paw centered near (37% width, 51% height) physically touching the viewer-left underside of the muzzle. The viewer-right front paw remains planted. Preserve both attached hind legs and the complete tail. A tiny pink tongue may touch the raised paw. The raised limb is living cat anatomy, never a detachable prop.

IDENTITY AND SURFACE LOCK: the raised paw and planted paw must share Image 2's local coat colors, soft outline, fur length, texture scale, lighting, and restrained toe relief. Join muzzle, paw, wrist, shoulder and chest seamlessly. No gray guide residue, achromatic recoloring, smooth tube, straight cylinder, uniform-width capsule, hard edge, brush fan, radial strands, detached object, duplicate paw, ghost limb, clothing, accessory, food, bowl, or extra subject. {fur_lock}

POSE CONTRACT: {qwen_first_frame_spec(pose)}
{action_first_frame_rules(pose)}

OUTPUT CONTRACT: return exactly one clean static 512x512 first frame. Keep the complete cat within 72% of canvas width and height with at least 12% clear green margin on every edge. Every non-cat pixel must be flat pure #00FF00. No floor, floor line, shadow, reflection, scenery, text, logo, or watermark. Return only the completed grooming keyframe."""


def build_canonical_grooming_pose_transfer_prompt(
    pose: str,
    *,
    profile_fur_lock: str = "",
) -> str:
    """Transfer canonical identity onto the frozen grooming action anatomy."""
    _sync_local_video_runtime(globals(), {"build_canonical_grooming_pose_transfer_prompt"})

    fur_lock = profile_fur_lock.strip() or (
        "Preserve Image 2's exact natural coat length, visible volume, strand "
        "direction, genuine tufts, and source-bounded silhouette."
    )
    return f"""CRITICAL ACTION-POSE IDENTITY TRANSFER - keep Image 1 anatomy, render the uploaded pet:
Image 1 is the sampler canvas and a deterministic high-key anatomy maquette derived from the frozen validated grooming action geometry. Its complete silhouette, head and torso placement, real viewer-left shoulder-to-elbow-to-wrist-to-paw anatomy, paw-to-mouth contact, viewer-right planted front paw, attached hind legs, tail, camera, scale, framing, and pure-green background are authoritative. Before this request all original RGB, chroma, raw luma, low-frequency markings, face detail, lighting, and fur texture were removed. Its broad low-contrast internal relief is a soft depth volume for the real raised foreleg: the upper rounded end is the paw touching the underside of the muzzle and the lower end joins the viewer-left shoulder and chest. It contains no internal green hole, line, discrete part color, fur marking, hanging item, cord, capsule, microphone, or rendering-style reference.
Image 2 is the approved Qwen full-body canonical generated from the uploaded identity view set. It is the sole target rendering authority. Replace every subject pixel from Image 1 with Image 2's exact recognizable face and head silhouette, green eyes, ears, nose, coat palette and localized markings, body proportions, natural coat category, clean fur surface, paws, tail, restrained stylized 3D material, and neutral-white lighting while preserving Image 1's action anatomy.
There is deliberately no third reference image in this recovery pass. Image 2 already consolidates the uploaded front, left, and right identity views. Do not infer a neutral sitting pose from the identity views that created Image 2, and do not replace Image 1's raised-paw action with Image 2's two-paws-down canonical pose.

ACTION FREEZE CONTRACT: Image 1 already contains a real furry foreleg and paw at the mouth; repaint that anatomy in place. Do not replace it with a hanging item, curved strip, strap, pendant, banana shape, tongue-shaped object, microphone, smooth tube, or detached prop. Do not return to a neutral two-paws-down sit and do not mirror the action. Exactly one viewer-left foreleg remains continuously attached from the visible shoulder near (38% width, 67% height), through a soft organic elbow and wrist, to one rounded furry paw near (37% width, 51% height) touching the viewer-left underside of the muzzle. The viewer-right front paw remains planted. Keep both attached hind legs and the complete tail. Preserve Image 1's subject scale, crop, camera, and action silhouette.

IDENTITY REPAINT CONTRACT: the final result must look exactly like Image 2, never like the grooming reference that supplied Image 1 geometry. Repaint the whole animal, including the face, torso, raised foreleg, planted paw, hind legs, and tail. The raised and planted paws must share the same canonical coat family, fur length, texture scale, lighting, and toe treatment. Join muzzle, paw, wrist, shoulder, and chest seamlessly. No residual anatomy-maquette gray, borrowed black-and-white action-reference markings, new dominant color, invented stripe, warm-brown/orange drift, cyan/blue/red/pink drift, hard object, brush fan, radial strands, duplicate limb, clothing, accessory, food, bowl, or extra subject. {fur_lock}

POSE CONTRACT: {qwen_first_frame_spec(pose)}
{action_first_frame_rules(pose)}

OUTPUT CONTRACT: return exactly one clean static 512x512 first frame. Keep the complete cat within 72% of canvas width and height with at least 12% clear green margin on every edge. Every non-cat pixel must be flat pure #00FF00. No floor, floor line, shadow, reflection, scenery, text, logo, or watermark. Return only the identity-correct grooming keyframe."""


def build_dog_eating_prompt_only_action_transfer_prompt(
    pose: str,
    *,
    profile_fur_lock: str = "",
) -> str:
    """Transfer white-model eating geometry onto the approved dog identity."""
    _sync_local_video_runtime(
        globals(), {"build_dog_eating_prompt_only_action_transfer_prompt"}
    )

    if action_key(pose) != "eating":
        raise ValueError("Prompt-only dog eating transfer supports only eating.")
    del profile_fur_lock
    return (
        "Image 1 is the only identity and appearance authority. Image 2 is a neutral "
        "white action-geometry guide and flat synthetic label, not a picture of the "
        "target dog: copy only "
        "its eating joints, camera, ground contact, muzzle-to-food contact, and bowl "
        "placement. Treat every white or gray animal pixel in Image 2 as a coordinate "
        "mask with zero color, value, material, texture, lighting, marking, breed, or "
        "identity meaning. Erase that synthetic surface instead of rendering it. "
        "Never copy Image 2's breed, skull, "
        "ears, muzzle, body proportions, tail, color, material, lighting, or "
        "surface. Edit Image 1. Keep this exact dog and its restrained stylized 3D "
        "identity unchanged. Preserve its exact apparent age and breed geometry: "
        "skull and cheek width, muzzle length/width/taper, ear topology/size/"
        "carriage, eyes, nose, coat colors and markings, natural coat length/"
        "volume, torso and leg proportions, paws, and the tail's exact thickness, "
        "shape and carriage. Change only the pose to active eating. Put exactly "
        "one small shallow matte bowl ON THE GROUND immediately between and "
        "touching the two planted front paws. Put only a small low level layer of "
        "ordinary brown dry kibble fully below and completely INSIDE the continuous "
        "bowl rim. Every kibble pellet must remain inside the bowl; no pellet, heap, "
        "ring, scatter, spill, overflow or food may appear outside the bowl. Lower "
        "and bend the dog's head and neck all the way down until the lower muzzle is "
        "physically inside and overlapped by the kibble and near rim. The dog eats "
        "from the ground bowl; it never lifts, carries, wears, bites or holds the "
        "bowl. Repose the torso and legs naturally as needed, keeping all four "
        "complete attached legs and paws and the complete tail visible. Do not "
        "change folded ears to upright ears or upright ears to folded ears, do not "
        "shorten or widen the muzzle, and do not change a straight tail to a curled "
        "tail or a curled tail to a straight tail. Keep one complete natural "
        "unclothed dog with no collar, accessory, extra subject or extra prop. Every "
        "background pixel stays flat pure #00FF00 green, with generous clearance "
        "and no floor, food, shadow, reflection, gradient, text or watermark."
    )


def build_dog_drinking_prompt_only_action_transfer_prompt(
    pose: str,
    *,
    profile_fur_lock: str = "",
) -> str:
    """Transfer white-model drinking geometry onto the approved dog identity."""
    _sync_local_video_runtime(
        globals(), {"build_dog_drinking_prompt_only_action_transfer_prompt"}
    )

    if action_key(pose) != "drinking":
        raise ValueError("Prompt-only dog drinking transfer supports only drinking.")
    del profile_fur_lock
    return (
        "Image 1 is the only identity and appearance authority. Image 2 is a neutral "
        "white action-geometry guide and flat synthetic label, not a picture of the "
        "target dog: copy only "
        "its drinking joints, camera, ground contact, head-to-water contact, and bowl "
        "placement. Treat every white or gray animal pixel in Image 2 as a coordinate "
        "mask with zero color, value, material, texture, lighting, marking, breed, or "
        "identity meaning. Erase that synthetic surface instead of rendering it. "
        "Never copy "
        "Image 2's breed, skull, ears, muzzle, body proportions, tail, color, "
        "material, lighting, or surface. Edit Image 1. Keep this exact dog and its "
        "restrained stylized 3D "
        "identity unchanged. Preserve its exact apparent age and breed geometry: "
        "skull and cheek width, muzzle length/width/taper, ear topology/size/"
        "carriage, eyes, nose, coat colors and markings, natural coat length/"
        "volume, torso and leg proportions, paws, and the tail's exact thickness, "
        "shape and carriage. Change only the pose to active drinking. Put exactly "
        "one small shallow water bowl ON THE GROUND immediately in front of the "
        "two planted front paws. Lower and bend the dog's head and neck until the "
        "lower muzzle physically reaches and slightly overlaps the bowl's near rim "
        "and one tiny anatomically attached pink tongue tip touches the water "
        "surface. Keep the water low and fully inside the bowl with only subtle "
        "ripples; no food, splash, spill, stream, drool, long tongue or panting "
        "smile. The bowl stays separate from the body and must not hide, replace, "
        "lift or fuse with a paw. Repose the torso and legs naturally as needed, "
        "keeping all four complete attached legs and paws and the complete tail "
        "visible. Do not change folded ears to upright ears or upright ears to "
        "folded ears, do not shorten or widen the muzzle, and do not change a "
        "straight tail to a curled tail or a curled tail to a straight tail. Keep "
        "one complete natural unclothed dog with no collar, accessory, extra "
        "subject or extra prop. Every background pixel stays flat pure #00FF00 "
        "green, with generous clearance and no floor, shadow, reflection, gradient, "
        "text or watermark."
    )


def build_dog_toileting_prompt_only_action_transfer_prompt(
    pose: str,
    *,
    profile_fur_lock: str = "",
) -> str:
    """Transfer white-model toileting geometry onto the approved dog identity."""
    _sync_local_video_runtime(
        globals(), {"build_dog_toileting_prompt_only_action_transfer_prompt"}
    )

    if action_key(pose) != "toileting":
        raise ValueError("Prompt-only dog toileting transfer supports only toileting.")
    del profile_fur_lock
    return (
        "Image 1 is the only identity and appearance authority. Image 2 is a neutral "
        "white action-geometry guide and flat synthetic label, not a picture of the "
        "target dog: copy only "
        "its seated toileting joints, camera, paw contact, and complete tray footprint. "
        "Treat every white or gray animal pixel in Image 2 as a coordinate mask with "
        "zero color, value, material, texture, lighting, marking, breed, or identity "
        "meaning. Erase that synthetic surface instead of rendering it. Never copy Image 2's "
        "breed, skull, ears, muzzle, body proportions, tail, color, material, "
        "lighting, or surface. Edit Image 1. Keep this exact dog and its restrained "
        "stylized 3D "
        "identity unchanged. Preserve its exact apparent age and breed geometry: "
        "skull and cheek width, muzzle length/width/taper, ear topology/size/"
        "carriage, eyes, nose, coat colors and markings, natural coat length/"
        "volume, torso and leg proportions, paws, and the tail's exact thickness, "
        "shape and carriage. Change only the pose to a stable natural seated "
        "toileting pose on exactly one clean low rectangular pet toilet tray. "
        "Place the tray UNDER the dog, never behind it: both front paws and the "
        "seated hindquarters physically rest on the tray's visible grid insert. "
        "The continuous outer rim must remain visible beyond the dog's silhouette "
        "on the left, right, front and rear, and the broad tray footprint must be "
        "wider than the seated dog. Keep the tray centered, complete, horizontal, "
        "fixed and fully inside the canvas. Repose the torso and legs naturally as "
        "needed while keeping all four complete attached legs and paws, both ears "
        "and the complete tail visible. Do not change folded ears to upright ears "
        "or upright ears to folded ears, do not shorten or widen the muzzle, and "
        "do not change a straight tail to a curled tail or a curled tail to a "
        "straight tail. Keep one complete natural unclothed dog with no collar, "
        "accessory, extra subject or extra prop. Show no urine, feces, litter, "
        "mess, duplicate tray or litter box. Every background pixel stays flat "
        "pure #00FF00 green, with generous clearance and no floor continuation, "
        "shadow, reflection, gradient, text or watermark."
    )


def build_advisory_grooming_action_transfer_prompt(
    pose: str,
    *,
    profile_fur_lock: str = "",
) -> str:
    """Return the verified direct single-image grooming edit instruction.

    The four-step distilled model followed this compact wording across four
    independent seeds. Repeating full pose/profile contracts made the edit less
    reliable, so Image 1 carries appearance context and the text states only the
    required topology change.
    """
    _sync_local_video_runtime(globals(), {"build_advisory_grooming_action_transfer_prompt"})

    del pose, profile_fur_lock
    return (
        "Edit Image 1. Replace the neutral two-front-paws-down sitting pose "
        "with a natural cat grooming pose. Move the front leg visible on the "
        "left side of the image from its current ground position up to the "
        "mouth. That same leg stays attached at the shoulder, bends naturally "
        "through the elbow and wrist, and ends in one rounded furry paw "
        "touching a tiny pink tongue at the mouth. Remove its former ground "
        "paw completely and fill the vacated outline with the same flat pure "
        "#00FF00 green background. Keep the front paw on the right side of the "
        "image planted. Exactly two front paws total: one at the mouth and one "
        "on the ground. Preserve the cat's exact face, body, coat colors, "
        "markings, fur, tail, camera, scale, framing, lighting, and flat "
        "pure-green background. Keep the complete cat visible. No floor or "
        "scenery."
    )


def build_advisory_grooming_identity_refine_prompt(
    pose: str,
    *,
    source_attempt: int,
    profile_fur_lock: str = "",
) -> str:
    """Repair detected face drift while keeping grooming geometry pixel-locked."""
    _sync_local_video_runtime(globals(), {"build_advisory_grooming_identity_refine_prompt"})

    fur_lock = profile_fur_lock.strip() or (
        "Preserve Image 2's exact natural coat length, visible volume, strand "
        "direction, genuine tufts, and source-bounded silhouette."
    )
    return f"""CONDITIONAL GROOMING IDENTITY REPAIR - MASKED UPPER-HEAD PASS (INTERIOR ONLY); the white maquette is absent from this pass:
IMAGE 1 is the normalized grooming action draft from attempt {source_attempt} and is the sampler canvas. Its complete seated pose, camera, framing, body placement, viewer-left shoulder-to-raised-paw path, paw-to-muzzle contact, viewer-right planted foreleg, attached hind legs, tail, body coat, existing head silhouette and every pixel outside the supplied upper-head edit mask are already accepted and must remain unchanged. The mask is strictly contained inside the existing upper-head foreground. It contains no green margin and gives no permission to enlarge, narrow, rotate or replace the head outline, cheeks or ears.
IMAGE 2 is the approved Qwen full-body canonical produced from the uploaded front and optional side identity views. It is the sole pet-identity authority. Use it only to restore the masked head and face: recognizable head silhouette, cheek width, eyes and eye color, ears, nose, muzzle, forehead fur and localized facial markings.
IMAGE 3 is an enlarged crop derived deterministically from the same Image 2 canonical. It is the strongest face-detail authority. Match its eye size and spacing, cheek contour, forehead width, ear proportions, nose and muzzle inside the mask. It introduces no second pet and no pose.

NO-MAQUETTE CONTRACT: no white-model, anatomy-maquette, profile animal, action-reference face, plastic surface or generic cat identity is present or permitted in this pass. Inside the masked head interior only, correct a detected smooth white-model face, black oval toy nose, bead-eye treatment, muzzle, facial-marking or palette drift wherever it conflicts with Images 2 and 3. Do not regenerate an already-correct face, change the source head pose, or average the draft face with a generic face.

MASKED ACTION FREEZE: edit only pixels enabled by the sampler's upper-head-interior noise mask. Do not repaint, erase, move or reinterpret the head outline, ears, cheek boundary, neck, chest, shoulder, raised foreleg, raised paw, planted foreleg, planted paw, hind legs, torso or tail. Never return to a neutral two-paws-down sit. Exactly one viewer-left foreleg remains continuously attached from shoulder through elbow and wrist to one rounded furry paw touching the underside of the muzzle. Exactly one viewer-right front paw remains planted. A single tiny pink tongue may touch the raised paw; there is no object in the mouth or paw.

FACE RESTORE: make the final animal immediately recognizable as Image 2 through the masked face. Preserve Image 2's cheek fullness, eye geometry and color, forehead, ears, nose, muzzle, facial markings and natural face fur. Blend the repaired head seamlessly into Image 1 at the mask edge; the accepted body and both paws remain Image 1 pixels. {fur_lock}

POSE CONTRACT: {qwen_first_frame_spec(pose)}

OUTPUT CONTRACT: return exactly one clean grooming first frame at the sampler size. Every non-cat pixel is flat pure #00FF00. No beige, white, gray or photographic background; no floor, floor line, shadow, reflection, text, logo, watermark, prop, duplicate limb or extra subject. Preserve every unmasked action pixel and the exact existing head silhouette while repairing only the detected canonical face drift inside the mask."""


def advisory_grooming_retry_anatomy_prompt(previous_issues: Iterable[str]) -> str:
    """Return concise positive anatomy and face feedback for advisory retries."""
    _sync_local_video_runtime(globals(), {"advisory_grooming_retry_anatomy_prompt"})

    issues = {str(issue) for issue in previous_issues}
    structural_issues = issues & GROOMING_ADVISORY_BLOCKING_POSE_ISSUES
    face_issues = issues & GROOMING_ADVISORY_MASKED_REPAIR_UNSAFE_ISSUES - {
        "raised_paw_hard_object_texture_signature"
    }
    if not structural_issues and not face_issues:
        return ""
    corrections: list[str] = []
    if issues & {
        "canonical_grooming_action_edit_too_weak",
        "original_planted_forepaw_topology_retained",
    }:
        corrections.append(
            "The previous candidate copied Image 1's neutral two-front-paws-down "
            "topology. This full-canvas retry must relocate, not duplicate, the "
            "viewer-left foreleg: remove its old ground paw and toe outline as "
            "anatomy, turn its exact vacated footprint pure #00FF00, and draw that "
            "same single foreleg continuously from shoulder to one paw at the "
            "mouth. Count exactly two front paws before returning: raised-left at "
            "the mouth and planted-right on the ground."
        )
    if structural_issues:
        corrections.append(
            "The previous candidate did not form the viewer-left raised foreleg "
            "as continuous living anatomy. Restart from Image 1 and use only the "
            "prompt; the full canvas is available for a coherent redraw. Rebuild "
            "that chest-to-mouth "
            "corridor with one "
            "connected shoulder, a softly bent elbow, a tapered wrist and one "
            "rounded furry paw touching the underside of the muzzle. Match the "
            "raised paw to the planted paw in coat, texture, lighting and toe form."
        )
    if face_issues:
        corrections.append(
            "The previous face drifted from the approved pet. Rebuild the complete "
            "head from Image 1 and the enlarged Image 2 crop: restore the same cheek "
            "width, eye size, eye color and spacing, ears, nose, muzzle and localized "
            "facial markings. No video frame or pose-reference animal is available "
            "or permitted as identity evidence."
        )
    corrections.append(
        "The corrected frame contains only the cat's natural anatomy and surrounding "
        "green negative space. Preserve the viewer-right planted foreleg, both hind "
        "legs, tail, framing and canonical identity."
    )
    return (
        f"GROOMING RETRY CORRECTION ({GROOMING_ADVISORY_RETRY_ANATOMY_POLICY}):\n"
        + " ".join(corrections)
    )


def advisory_grooming_retry_requires_palette_correction(
    previous_issues: Iterable[str],
) -> bool:
    """Apply the palette retry sentence only after an observed palette issue."""
    _sync_local_video_runtime(globals(), {"advisory_grooming_retry_requires_palette_correction"})

    return any(
        any(
            token in str(issue)
            for token in ("palette", "chromatic", "achromatic", "grooming_face")
        )
        for issue in previous_issues
    )


def advisory_grooming_candidate_safe_for_masked_repair(
    minimum_gate: dict[str, Any] | None,
) -> bool:
    """Do not freeze a visibly non-anatomical raised-paw region into retry 3."""
    _sync_local_video_runtime(globals(), {"advisory_grooming_candidate_safe_for_masked_repair"})

    if not isinstance(minimum_gate, dict):
        return False
    issues = {str(issue) for issue in minimum_gate.get("issues", [])}
    return not bool(issues & GROOMING_ADVISORY_MASKED_REPAIR_UNSAFE_ISSUES)


def advisory_grooming_candidate_safe_for_identity_refine(
    minimum_gate: dict[str, Any] | None,
) -> bool:
    """Allow only an action-safe draft with detected, head-repairable drift."""
    _sync_local_video_runtime(globals(), {"advisory_grooming_candidate_safe_for_identity_refine"})

    if not isinstance(minimum_gate, dict):
        return False
    if minimum_gate.get("available") is not True:
        return False
    issues = {str(issue) for issue in minimum_gate.get("issues", [])}
    if not issues:
        # A draft that already passed the identity contract must be selected
        # byte-for-byte. Running a second generative face pass can only add
        # identity variance and was the source of the v2.24.6 regression.
        return False
    if not issues.issubset(
        GROOMING_ADVISORY_IDENTITY_REFINE_SOURCE_ALLOWED_ISSUES
    ):
        return False
    pose_quality = minimum_gate.get("pose")
    if isinstance(pose_quality, dict) and pose_quality.get("acceptable") is not True:
        return False
    return True


def build_canonical_grooming_full_pose_identity_refine_prompt(
    pose: str,
    *,
    source_attempt: int,
    profile_fur_lock: str = "",
) -> str:
    """Repaint a structurally proven grooming candidate with canonical identity."""
    _sync_local_video_runtime(globals(), {"build_canonical_grooming_full_pose_identity_refine_prompt"})

    fur_lock = profile_fur_lock.strip() or (
        "Preserve Image 2's exact natural coat length, visible volume, strand "
        "direction, genuine tufts, and source-bounded silhouette."
    )
    return f"""CRITICAL POSE-SAFE IDENTITY REFINE - freeze the proven grooming action and repaint the pet:
Image 1 is the normalized grooming candidate from attempt {source_attempt} and is the sampler canvas. Its matte was reconstructed only by the union of the semantic pet matte and the digest-bound deterministic action seed. It then passed the independent complete-subject check, strict raised-foreleg pose gate, guide alignment, organic shoulder-to-paw structure, paw-to-muzzle contact, and fur-surface gate. Its complete outer silhouette, head and torso placement, viewer-left shoulder-to-elbow-to-wrist-to-paw path, viewer-right planted front paw, both attached hind legs, tail, scale, framing, camera, and pure-green negative space are authoritative. Its warm/off-model coat palette, markings, face detail, lighting and material are not identity authorities.
Image 2 is the approved Qwen full-body canonical generated from the uploaded front, left and right identity views. It is the sole identity and target-rendering authority. Repaint every animal pixel from Image 2: exact recognizable face and head silhouette, green eyes, ears, nose, achromatic coat-value balance, localized markings, body proportions, natural coat category, clean fur surface, paws, tail, restrained stylized 3D material, and neutral-white lighting.
Image 3 is the deterministic high-key action-anatomy seed whose digest was used to validate Image 1's matte. It corroborates geometry only: the same complete seated silhouette, broad soft raised-foreleg depth volume, real viewer-left shoulder-to-elbow-to-wrist-to-paw anatomy, paw-to-mouth contact, viewer-right planted paw, attached hind legs, tail, scale, and framing. It contains no internal green hole or discrete part color; its neutral distance/depth shading is not fur, markings, identity, material, lighting, or style. Never average Image 3's tone into Image 2.

ACTION FREEZE CONTRACT: Image 1 already contains a real attached furry foreleg and paw at the mouth; repaint that anatomy in place. Image 3 confirms the same action but may not replace Image 1's pixels or silhouette. Do not return to a neutral two-paws-down sit, mirror the action, crop the body, or change camera and scale. Exactly one viewer-left foreleg remains continuously attached from the visible shoulder near (38% width, 67% height), through a soft organic elbow and wrist, to one rounded furry paw near (37% width, 51% height) touching the viewer-left underside of the muzzle. The viewer-right front paw remains planted. Keep both attached hind legs and the complete tail. Never turn the raised limb into a hanging item, curved strip, strap, pendant, tongue-shaped object, microphone, smooth tube, detached prop, or extra paw.

IDENTITY REPAINT CONTRACT: replace the whole animal appearance, including face, torso, raised foreleg, planted paw, hind legs, and tail, with Image 2's identity while retaining Image 1's proven action geometry. The raised and planted paws must share Image 2's coat family, fur length, texture scale, lighting, and restrained toe treatment. Join muzzle, paw, wrist, shoulder, and chest seamlessly. No residual warm brown/orange cast, achromatic balance loss, borrowed action-reference markings, anatomy-seed gray, cyan/blue/red/pink drift, invented stripe, hard object, brush fan, radial strands, duplicate limb, clothing, accessory, food, bowl, or extra subject. {fur_lock}

POSE CONTRACT: {qwen_first_frame_spec(pose)}
{action_first_frame_rules(pose)}

OUTPUT CONTRACT: return exactly one clean static 512x512 first frame. Keep the complete cat within 72% of canvas width and height with at least 12% clear green margin on every edge. Every non-cat pixel must be flat pure #00FF00. No floor, floor line, shadow, reflection, scenery, text, logo, or watermark. Return only the identity-correct grooming keyframe."""


def build_canonical_eating_identity_refine_prompt(
    pose: str,
    *,
    source_attempt: int,
    profile_fur_lock: str = "",
) -> str:
    """Preserve an approved eating layout while repainting canonical identity."""
    _sync_local_video_runtime(globals(), {"build_canonical_eating_identity_refine_prompt"})

    fur_lock = profile_fur_lock.strip() or (
        "Preserve Image 2's exact natural coat length, visible volume, strand "
        "direction, genuine tufts, and source-bounded silhouette."
    )
    return f"""CRITICAL EATING IDENTITY REFINE - preserve the approved action, repaint only the pet identity:
Image 1 is the normalized eating candidate from attempt {source_attempt} and is the sampler canvas. Its complete crouched body layout, head height and angle, lower-muzzle-to-kibble overlap, both planted front paws, attached rear legs, tail, single bowl, kibble placement, subject scale, framing, and pure-green negative space are authoritative. Preserve this exact action and prop geometry. Image 1's pet colors, markings, face details, fur texture, lighting, and rendering style are not authoritative and may contain a warm, cool, dark, pale, or otherwise off-model identity drift. In particular, any stronger, darker, wider, more numerous, or more extensive stripes in Image 1 are rejected artifacts rather than action geometry.
Image 2 is the approved Qwen full-body canonical and is the target rendering and primary identity authority for every pet pixel. Repaint the entire pet from Image 2: exact head silhouette, eye color and spacing, ears, nose, source-faithful coat color and value balance, actual localized markings, body proportions, natural fur category and surface, paws, tail, restrained stylized 3D material, and neutral-white lighting. Do not copy any hue, value, marking, or lighting cast from Image 1 that is absent from Image 2 and Image 3.
Image 3 is the original uploaded front photograph of this same pet. It corroborates immutable real identity only: face proportions, eye color, nose, coat-value balance, actual marking placement, coat length and paw color. It has no eating-pose, bowl, crop, background, room lighting, photographic-material, or output-style authority. Image 1 remains the sole eating and bowl geometry authority; Image 2 remains the sole target-rendering authority. Do not copy the room, floor, shadow, or photographic background from Image 3.

SPECIES AND FACE CONTRACT: the species, breed type, head silhouette and facial proportions from Images 2 and 3 are immutable. A cat must remain a cat and a dog must remain a dog. Preserve the exact pointed or folded ear topology, ear size and placement, eye spacing, cheek width, nose size, muzzle length and muzzle width. Never turn the pet into a bear, polar bear, ferret, otter, rodent or another breed; never invent round bear ears, an enlarged black nose, an oversized dark muzzle, a mask-like face, or a barrel-shaped substitute body.

MARKING CORRECTION CONTRACT: use Image 3 to decide which real stripes, spots, patches and dark facial bands exist and how subtle they are; use Image 2 for their stylized rendering. Repaint over every amplified marking inherited from Image 1 while leaving its silhouette and eating geometry fixed. A faint source stripe must remain faint and localized, not become a high-contrast torso ring or repeated flank bar. If Images 2 and 3 do not corroborate a marking, remove it completely using the nearest genuine base-coat color. Never average Image 1's marking density or contrast into the result.

ACTION FREEZE CONTRACT: keep exactly one small shallow light-beige matte bowl with brown kibble. Keep both complete front paws visibly planted beside the bowl. The lower muzzle and mouth must remain physically inside the kibble with 6 to 10 output pixels of visible overlap; the near rim or kibble may partly occlude the nose and lower muzzle. There must be no green, white, or background gap between mouth and food. Never raise the head, move the bowl away, turn eating into sniffing, let a paw hold food, add a second bowl, or let the bowl hide or replace a limb.

IDENTITY REPAINT CONTRACT: change the pet appearance everywhere while preserving Image 1's action silhouette. The bowl and kibble are props, so retain their light-beige and brown colors without transferring those warm colors into the pet. Remove every star, flower, decorative mark, floor, floor line, shadow and scenery already excluded by the normalized sampler. Preserve clean canonical fur with no radial brush fan, long artificial strands, smooth plastic, new marking, new dominant hue, clothing, accessory, or extra subject. {fur_lock}

POSE CONTRACT: {qwen_first_frame_spec(pose)}
{action_first_frame_rules(pose)}

OUTPUT CONTRACT: return exactly one clean static 512x512 first frame. Keep the complete pet and its single bowl within 72% of canvas width and height with at least 12% clear green margin on every edge. Every non-pet, non-bowl pixel must be flat pure #00FF00. No floor, floor line, shadow, reflection, scenery, text, logo, or watermark. Return only the identity-correct eating keyframe."""


def build_canonical_eating_contact_repair_prompt(
    pose: str,
    *,
    source_attempt: int,
    source_gap_px: int,
    prealignment_shift_px: int,
    prealignment_expected_seed_gap_px: int,
    profile_fur_lock: str = "",
) -> str:
    """Close a measured muzzle-to-kibble gap without redrawing the body."""
    _sync_local_video_runtime(
        globals(),
        {"build_canonical_eating_contact_repair_prompt"},
    )

    fur_lock = profile_fur_lock.strip() or (
        "Preserve Image 2's exact natural coat length, visible volume, strand "
        "direction, genuine tufts, and source-bounded silhouette."
    )
    measured_gap = max(0, int(source_gap_px))
    prealigned_shift = max(0, int(prealignment_shift_px))
    expected_seed_gap = max(0, int(prealignment_expected_seed_gap_px))
    return f"""CRITICAL LOCAL EATING-CONTACT REPAIR - close the measured air gap and change nothing else:
Image 1 is the identity-safe normalized eating candidate from attempt {source_attempt} and is the sampler canvas. Its head and neck have already been deterministically pre-aligned downward by {prealigned_shift} pixels from the measured {measured_gap}-pixel source gap, targeting an approximately {expected_seed_gap}-pixel seed gap while the bowl and lower body remained byte-stable. This pre-aligned head height is authoritative: preserve it, fuse its local seams, and never lift the head back toward the rejected source position. Its complete body, both planted front paws, attached rear legs, tail, single bowl, kibble, scale, framing, camera, pure-green negative space, coat palette, markings, and rendering are already approved and must remain pixel-stable outside the supplied local noise mask.
Image 2 is the approved Qwen full-body canonical generated from the uploaded identity views. It only corroborates the exact pet identity, face, coat, markings, fur category, material, and lighting while the small masked lower-muzzle seam is fused.
Image 3 is the neutral Main_case white-maquette eating geometry. It has no identity, face, eye, coat, color, texture, material, lighting, or style authority. Use it only to confirm that the whole head and neck lower coherently until the mouth reaches the fixed food surface.

LOCAL EDIT CONTRACT: edit only inside the supplied feathered lower-muzzle and mouth-to-kibble seam mask. The pre-aligned whole head, eyes, ears, cheeks, forehead, neck, torso and coat markings are already final and must not be redrawn. Fuse only the lower muzzle edge without stretching the lips, mouth, chin, or nose. The lower muzzle must touch or enter the brown kibble so the final measured vertical gap is 0 to 8 output pixels, preferably about 4 pixels. A small near-rim or kibble occlusion over the nose or lower muzzle is required evidence of eating. Keep the former head location pure #00FF00 where exposed.

PIXEL LOCK CONTRACT: never move, resize, repaint, recolor, replace, duplicate, or erase the bowl, kibble, either planted front paw, rear legs, torso, back, tail, eyes, ears, upper face, canvas boundary, or camera. After generation the system restores every pixel outside the lower-muzzle mask, plus every bowl-and-lower-body pixel, from Image 1; do not rely on changing those locked regions. Do not lift the bowl toward the face, slide it sideways, enlarge it, add a second bowl, hide a paw, or turn eating into sniffing. Keep exactly one pet and one small shallow light-beige bowl. Preserve Image 1's identity-safe coat and Image 2's canonical facial identity with no new stripe, patch, hue, accessory, floor, shadow, reflection, text, logo, or watermark. {fur_lock}

POSE CONTRACT: {qwen_first_frame_spec(pose)}
{action_first_frame_rules(pose)}

OUTPUT CONTRACT: return exactly one clean static 512x512 first frame. Every non-pet, non-bowl pixel is flat pure #00FF00. Return only the locally repaired eating keyframe."""


def build_wan_prompt(base_prompt: str, pose: str, transparent_gif: bool = False) -> str:
    _sync_local_video_runtime(globals(), {"build_wan_prompt"})
    background_prompt = (
        "Keep the background perfectly uniform pure #00FF00 chroma-key green in every frame, with no shadow, gradient, texture, flicker, color shift, or green spill on the subject."
        if transparent_gif
        else "Preserve the background from the first frame."
    )
    if transparent_gif:
        motion_rules = action_video_rules(pose)
        subject_position_prompt = (
            "Keep the complete character and the single action-required bowl centered "
            "at the same screen position, scale, and perspective."
            if action_uses_bowl(pose)
            else "Keep the complete character centered at the same screen position, "
            "scale, and perspective. No prop may appear."
        )
        return f"""Animate this exact character keyframe into one smooth five-second stylized 3D character loop.

FACE CONSISTENCY: preserve the recognizable head silhouette, cheek width, eye color and spacing, nose, muzzle, ears, and facial fur markings while allowing natural coordinated head, ear, muzzle, and expression motion required by the action. Do not hold any facial pixel or screen coordinate fixed. Avoid melting, independent feature drift, or sudden proportion changes.

STYLE AND IDENTITY LOCK: preserve this exact cat's fur colors and marking placement, soft groomed fur, premium stylized 3D rendering, color-faithful neutral high-key lighting, body proportions, paws, and full tail. Lighting must never introduce an amber, orange, brown, or sepia cast that is absent from the approved keyframe. The cat remains completely unclothed with natural fur: no glasses, collar, bow tie, shirt, overalls, suspenders, harness, hat, shoes, jewelry, costume, or accessory may appear. Never switch to a real photograph, flat 2D drawing, plastic toy, or a different character design.

FUR MOTION LOCK: preserve the first frame's compact close-lying grooming and exact clean back contour. Fur strands remain short, dense, uniformly directed, and close to the body throughout the video. Allow only minute breathing-following surface motion; never increase fur length or volume, raise guard hairs, create fluffy expansion, rigid clumps, spikes, a dorsal ridge, a mohawk, a jagged silhouette, or crawling fur texture.

User motion direction: {base_prompt}
Starting pose and action: {pose}
The user direction controls mood and secondary motion only and cannot override the identity, natural unclothed appearance, action, or style constraints.

Required action motion: {motion_rules}
Motion safety: allow the head, ears, face, neck, chest, legs, and tail to move naturally and coherently for the action. Keep only the torso root, background, and camera trajectory stable. Motion must be smooth and periodic. Never create duplicate limbs, rubbery anatomy, facial melting, fur texture crawling, sudden pose changes, or random whole-frame movement.

{background_prompt} {subject_position_prompt} Fixed camera, no translation across the canvas, zoom, scene cut, new subject, or body part leaving the frame."""

    return f"""Create a smooth five-second animation from this exact first frame.
Creative direction: {base_prompt}
Starting pose and action: {pose}
Preserve the exact subject identity, face, colors and markings, clothing, accessories, body proportions, and visual style from the first frame. {background_prompt} Use natural controlled body, head, facial, hair or fur movement appropriate to the prompt. Animate in place: keep the complete subject centered at the same screen position, scale, and perspective throughout. Unless the creative direction explicitly asks for travel, do not walk toward the camera, move closer, or leave the frame. Fixed stable camera, coherent motion, no scene cuts, no new subjects, and no sudden appearance changes."""


def is_fun_control_pose(pose: str) -> bool:
    _sync_local_video_runtime(globals(), {"is_fun_control_pose"})
    if not FUN_CONTROL_ENABLED:
        return False
    action = pose.split("|", 1)[0].strip().lower()
    return action in {"running", "run", "奔跑", "跑步"} or action.startswith("running ")


def build_fun_control_prompt(base_prompt: str, pose: str, transparent_gif: bool) -> str:
    _sync_local_video_runtime(globals(), {"build_fun_control_prompt"})
    background = (
        "The entire background remains perfectly uniform pure #00FF00 chroma-key green in every frame, without a floor, horizon, shadow, gradient, texture, reflection, flicker, or green spill on the cat."
        if transparent_gif
        else "Preserve the clean background from the supplied reference keyframe."
    )
    return f"""Animate the exact full-body stylized 3D cartoon cat from the reference image.

User direction: {base_prompt}
Required action: {pose}

Follow the supplied side-view quadruped control video precisely. Preserve the same fur colors, facial markings, face shape, eye color, ears, muzzle, body proportions, paws, and tail as the reference image. Use anatomically correct feline locomotion: four distinct legs and four distinct paws, coherent shoulder and hip motion, correct paw contacts, natural spine compression and extension, and natural tail counterbalance. The control sequence determines the leg positions and contact timing. Never invent an independent gait or random paw motion.

Keep the recognizable face structurally stable while allowing natural head motion. The cat remains completely unclothed, with no collar, glasses, harness, clothing, shoes, or accessories. Keep the subject centered at a stable scale with a fixed side-view camera. {background} High temporal consistency, soft groomed fur, clean edges, no text, and no watermark."""


def qwen_recovery_profiles(
    *,
    user_lora: str | None,
    user_lora_strength: float,
    user_lora_mode: str,
    steps: int,
    cfg: float,
    denoise: float,
    user_loras: list[dict[str, Any]] | None = None,
    lock_strengths: bool = False,
) -> list[dict[str, Any]]:
    _sync_local_video_runtime(globals(), {"qwen_recovery_profiles"})
    effective_user_loras = normalize_user_lora_specs(
        user_loras,
        legacy_model_name=user_lora,
        legacy_strength=user_lora_strength,
        resolve_models=False,
    )
    reduced_user_loras = reduced_user_lora_specs(effective_user_loras)
    requested_total_strength = sum(
        float(item["strength"]) for item in effective_user_loras
    )
    reduced_total_strength = sum(float(item["strength"]) for item in reduced_user_loras)
    reduced_scale = (
        reduced_total_strength / requested_total_strength
        if requested_total_strength > 0.0
        else 1.0
    )
    final_user_loras = (
        [dict(item) for item in effective_user_loras]
        if lock_strengths
        else reduced_user_loras
    )
    final_scale = 1.0 if lock_strengths else reduced_scale
    final_name = (
        "requested_final_seed_retry" if lock_strengths else "safe_reduced_user_lora"
    )
    final_mode = user_lora_mode if lock_strengths else "stack"

    def legacy_fields(specs: list[dict[str, Any]]) -> dict[str, Any]:
        first = specs[0] if specs else None
        return {
            "user_lora": first["model_name"] if first else None,
            "user_lora_strength": first["strength"] if first else 0.0,
        }

    profiles = [
        {
            "name": "requested",
            "user_loras": [dict(item) for item in effective_user_loras],
            **legacy_fields(effective_user_loras),
            "user_lora_strength_scale": 1.0,
            "user_lora_mode": user_lora_mode,
            "steps": steps,
            "cfg": cfg,
            "denoise": denoise,
            "use_user_prompt": True,
        },
        {
            "name": "requested_seed_retry",
            "user_loras": [dict(item) for item in effective_user_loras],
            **legacy_fields(effective_user_loras),
            "user_lora_strength_scale": 1.0,
            "user_lora_mode": user_lora_mode,
            "steps": max(6, steps),
            "cfg": cfg,
            "denoise": 1.0,
            "use_user_prompt": True,
        },
        {
            "name": final_name,
            "user_loras": final_user_loras,
            **legacy_fields(final_user_loras),
            "user_lora_strength_scale": final_scale,
            "user_lora_mode": final_mode,
            "steps": max(8, min(10, steps)),
            "cfg": 1.0,
            "denoise": 1.0,
            "use_user_prompt": True,
        },
    ]
    return profiles[:QWEN_CHROMA_MAX_ATTEMPTS]


def qwen_pose_profiles(
    *,
    user_lora: str | None,
    user_lora_strength: float,
    user_lora_mode: str,
    steps: int,
    cfg: float,
    denoise: float,
    user_loras: list[dict[str, Any]] | None = None,
    first_total_cap: float | None = None,
    first_per_lora_cap: float | None = None,
    lock_strengths: bool = False,
) -> list[dict[str, Any]]:
    """Build progressively less pose-locking profiles for every image/video request."""
    _sync_local_video_runtime(globals(), {"qwen_pose_profiles"})
    requested = normalize_user_lora_specs(
        user_loras,
        legacy_model_name=user_lora,
        legacy_strength=user_lora_strength,
        resolve_models=False,
    )
    requested_total = sum(float(item["strength"]) for item in requested)
    effective_first_total_cap = (
        QWEN_POSE_USER_LORA_TOTAL_CAP
        if first_total_cap is None
        else float(first_total_cap)
    )
    effective_first_per_lora_cap = (
        QWEN_POSE_USER_LORA_PER_LORA_CAP
        if first_per_lora_cap is None
        else float(first_per_lora_cap)
    )
    if lock_strengths:
        configurations = [
            (
                "pose_capped",
                effective_first_total_cap,
                effective_first_per_lora_cap,
                user_lora_mode,
                steps,
            ),
            (
                "pose_exact_seed_retry",
                effective_first_total_cap,
                effective_first_per_lora_cap,
                user_lora_mode,
                max(6, min(8, steps)),
            ),
            (
                "pose_exact_final_retry",
                effective_first_total_cap,
                effective_first_per_lora_cap,
                user_lora_mode,
                max(8, min(10, steps)),
            ),
        ]
    else:
        configurations = [
            (
                "pose_capped",
                effective_first_total_cap,
                effective_first_per_lora_cap,
                user_lora_mode,
                steps,
            ),
            (
                "pose_reduced_retry",
                QWEN_POSE_USER_LORA_RETRY_TOTAL_CAP,
                min(
                    QWEN_POSE_USER_LORA_PER_LORA_CAP,
                    QWEN_POSE_USER_LORA_RETRY_TOTAL_CAP,
                ),
                "stack",
                max(6, min(8, steps)),
            ),
            (
                "pose_minimal_retry",
                QWEN_POSE_USER_LORA_FINAL_TOTAL_CAP,
                min(
                    QWEN_POSE_USER_LORA_PER_LORA_CAP,
                    QWEN_POSE_USER_LORA_FINAL_TOTAL_CAP,
                ),
                "stack",
                max(8, min(10, steps)),
            ),
        ]
    profiles: list[dict[str, Any]] = []
    for quality_retry, (
        name,
        total_cap,
        per_lora_cap,
        mode,
        effective_steps,
    ) in enumerate(
        configurations,
        start=1,
    ):
        effective_loras, scale = cap_user_lora_specs(
            requested,
            total_cap=total_cap,
            per_lora_cap=per_lora_cap,
        )
        first = effective_loras[0] if effective_loras else None
        profiles.append(
            {
                "name": name,
                "user_loras": effective_loras,
                "user_lora": first["model_name"] if first else None,
                "user_lora_strength": first["strength"] if first else 0.0,
                "user_lora_strength_scale": (scale if requested_total > 0.0 else 1.0),
                "user_lora_mode": mode,
                "steps": effective_steps,
                "cfg": max(1.0, cfg),
                "denoise": 1.0 if quality_retry > 1 else denoise,
                "use_user_prompt": False,
                "quality_retry": quality_retry,
                "total_strength_cap": total_cap,
                "per_lora_strength_cap": per_lora_cap,
            }
        )
    return profiles[:QWEN_POSE_MAX_ATTEMPTS]


def pose_generation_profiles(
    pose: str,
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add action-specific seed retries for difficult running and bowl keyframes."""
    _sync_local_video_runtime(globals(), {"pose_generation_profiles"})
    selected = [
        {
            **profile,
            "user_loras": [dict(item) for item in profile.get("user_loras", [])],
        }
        for profile in profiles
    ]
    if not selected:
        return selected
    action = action_key(pose)
    if action == "running":
        extra_attempts = QWEN_RUNNING_POSE_EXTRA_ATTEMPTS
        retry_name = "running_ground_contact_seed_retry"
    elif action_uses_bowl(pose):
        extra_attempts = QWEN_BOWL_POSE_EXTRA_ATTEMPTS
        retry_name = QWEN_BOWL_POSE_RETRY_NAME
    else:
        return selected
    # Once a bowl attempt has reached the minimal LoRA profile, keep that safer
    # coat/pose influence for the extra seed retry instead of jumping back to the
    # stronger first profile that may have caused fur expansion.
    retry_source = selected[-1]
    for extra_index in range(extra_attempts):
        repeated = {
            **retry_source,
            "user_loras": [dict(item) for item in retry_source.get("user_loras", [])],
        }
        repeated["name"] = f"{retry_name}_{extra_index + 1}"
        repeated["quality_retry"] = max(
            4,
            int(selected[-1]["quality_retry"]) + 1,
        )
        repeated["steps"] = max(10, int(repeated["steps"]))
        selected.append(repeated)
    return selected


def action_candidate_generation_profiles(
    pose: str,
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize every action to the fixed five-candidate selection budget."""
    _sync_local_video_runtime(globals(), {"action_candidate_generation_profiles"})
    selected = [
        {
            **profile,
            "user_loras": [dict(item) for item in profile.get("user_loras", [])],
        }
        for profile in profiles[:QWEN_ACTION_CANDIDATE_ATTEMPTS]
    ]
    if not selected:
        raise RuntimeError("Action candidate generation requires at least one profile.")
    retry_source = selected[-1]
    action = action_key(pose) or "action"
    while len(selected) < QWEN_ACTION_CANDIDATE_ATTEMPTS:
        repeated = {
            **retry_source,
            "user_loras": [
                dict(item) for item in retry_source.get("user_loras", [])
            ],
        }
        next_attempt = len(selected) + 1
        repeated["name"] = f"{action}_best_of_five_seed_retry_{next_attempt}"
        repeated["quality_retry"] = next_attempt
        repeated["steps"] = max(10, int(repeated.get("steps", 10)))
        selected.append(repeated)
    return selected


def canonical_grooming_generation_profiles(
    pose: str,
    profiles: list[dict[str, Any]],
    *,
    enabled: bool,
) -> list[dict[str, Any]]:
    """Append finite full-pose and identity-refine recovery attempts."""
    _sync_local_video_runtime(globals(), {"canonical_grooming_generation_profiles"})

    selected = [
        {
            **profile,
            "user_loras": [dict(item) for item in profile.get("user_loras", [])],
        }
        for profile in profiles
    ]
    if not enabled or action_key(pose) != "grooming" or not selected:
        return selected
    selected = selected[:GROOMING_QWEN_FORELEG_REPAIR_ATTEMPT]
    while len(selected) < GROOMING_QWEN_FORELEG_REPAIR_ATTEMPT:
        repeated = {
            **selected[-1],
            "user_loras": [
                dict(item) for item in selected[-1].get("user_loras", [])
            ],
        }
        repeated["name"] = f"canonical_grooming_seed_retry_{len(selected) + 1}"
        repeated["quality_retry"] = len(selected) + 1
        selected.append(repeated)
    retry_source = selected[-1]
    for attempt, name in (
        (
            GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT,
            "canonical_palette_action_pose_transfer_primary",
        ),
        (
            GROOMING_CANONICAL_FULL_POSE_REFINE_ATTEMPT,
            "canonical_palette_action_pose_transfer_seed_retry",
        ),
    ):
        repeated = {
            **retry_source,
            "user_loras": [dict(item) for item in retry_source.get("user_loras", [])],
        }
        repeated.update(
            name=name,
            quality_retry=attempt,
            steps=max(
                GROOMING_CANONICAL_FULL_POSE_RESCUE_STEPS,
                int(repeated["steps"]),
            ),
            denoise=(
                GROOMING_CANONICAL_FULL_POSE_RESCUE_DENOISE
                if attempt == GROOMING_CANONICAL_FULL_POSE_RESCUE_ATTEMPT
                else GROOMING_CANONICAL_FULL_POSE_REFINE_DENOISE
            ),
        )
        selected.append(repeated)
    return selected


def advisory_grooming_generation_profiles(
    pose: str,
    profiles: list[dict[str, Any]],
    *,
    enabled: bool,
) -> list[dict[str, Any]]:
    """Build the finite raw-Qwen action-draft budget.

    A conditional generative identity-refine profile is appended only when its
    explicitly configured budget is non-zero.  Production keeps that budget at
    zero so the selected first frame is an untouched Qwen action-pass output.
    """
    _sync_local_video_runtime(globals(), {"advisory_grooming_generation_profiles"})

    selected = [
        {
            **profile,
            "user_loras": [dict(item) for item in profile.get("user_loras", [])],
        }
        for profile in profiles[:GROOMING_ADVISORY_ACTION_TRANSFER_MAX_ATTEMPTS]
    ]
    if not enabled or action_key(pose) != "grooming" or not selected:
        return selected
    retry_source = selected[-1]
    while len(selected) < GROOMING_ADVISORY_ACTION_TRANSFER_MAX_ATTEMPTS:
        repeated = {
            **retry_source,
            "user_loras": [dict(item) for item in retry_source.get("user_loras", [])],
        }
        repeated["name"] = f"grooming_action_draft_seed_retry_{len(selected) + 1}"
        repeated["quality_retry"] = len(selected) + 1
        repeated["steps"] = max(
            GROOMING_ADVISORY_ACTION_TRANSFER_STEPS,
            int(repeated["steps"]),
        )
        selected.append(repeated)
    for refine_index in range(GROOMING_ADVISORY_IDENTITY_REFINE_MAX_ATTEMPTS):
        identity_refine_profile = {
            **selected[-1],
            "user_loras": [
                dict(item) for item in selected[-1].get("user_loras", [])
            ],
        }
        identity_refine_profile.update(
            name="grooming_conditional_identity_refine",
            quality_retry=(
                GROOMING_ADVISORY_ACTION_TRANSFER_MAX_ATTEMPTS
                + refine_index
                + 1
            ),
            steps=max(
                GROOMING_ADVISORY_IDENTITY_REFINE_STEPS,
                int(identity_refine_profile["steps"]),
            ),
            denoise=GROOMING_ADVISORY_IDENTITY_REFINE_DENOISE,
        )
        selected.append(identity_refine_profile)
    if len(selected) != GROOMING_ADVISORY_TOTAL_ATTEMPTS:
        raise RuntimeError(
            "Advisory grooming profile count does not match the configured raw "
            "action and identity-refine budgets."
        )
    return selected


def resolve_fur_surface_retry_decision(
    *,
    action: str,
    rejection_streak: int,
    attempt: int,
    active_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Preserve the finite eating-only bowl recovery before failing closed."""
    _sync_local_video_runtime(globals(), {"resolve_fur_surface_retry_decision"})

    remaining_profile_names = [
        str(profile.get("name") or "") for profile in active_profiles[attempt:]
    ]
    recovery_prefix = f"{QWEN_BOWL_POSE_RETRY_NAME}_"
    pending_eating_bowl_recovery = bool(
        action == "eating"
        and remaining_profile_names
        and remaining_profile_names[0].startswith(recovery_prefix)
    )
    rejection_threshold_reached = bool(
        rejection_streak >= FUR_SURFACE_GATE_MAX_REJECTIONS
    )
    full_attempt_budget_completed = bool(attempt >= len(active_profiles))
    return {
        "policy": FUR_SURFACE_GATE_RETRY_POLICY,
        "action": action,
        "attempt": attempt,
        "attempt_budget": len(active_profiles),
        "rejection_streak": rejection_streak,
        "max_consecutive_rejections": FUR_SURFACE_GATE_MAX_REJECTIONS,
        "threshold_reached": rejection_threshold_reached,
        "remaining_profiles": remaining_profile_names,
        "next_profile": (
            remaining_profile_names[0] if remaining_profile_names else None
        ),
        "pending_eating_bowl_recovery": pending_eating_bowl_recovery,
        "full_attempt_budget_completed": full_attempt_budget_completed,
        "short_circuit_due": bool(
            rejection_threshold_reached
            and full_attempt_budget_completed
            and not pending_eating_bowl_recovery
        ),
        "blocks_scail_on_exhaustion": True,
    }


def pose_recovery_profiles(
    profiles: list[dict[str, Any]],
    minimum_profile_index: int,
) -> list[dict[str, Any]]:
    _sync_local_video_runtime(globals(), {"pose_recovery_profiles"})
    selected = [dict(profile) for profile in profiles[minimum_profile_index:]]
    if not selected:
        selected = [dict(profiles[-1])]
    while len(selected) < QWEN_POSE_MAX_ATTEMPTS:
        repeated = dict(selected[-1])
        repeated["name"] = f"{repeated['name']}_seed_retry_{len(selected) + 1}"
        selected.append(repeated)
    return selected[:QWEN_POSE_MAX_ATTEMPTS]


def qwen_workflow(
    input_name: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    prefix: str,
    user_lora: str | None = None,
    user_lora_strength: float = 1.0,
    user_lora_mode: str = "replace",
    steps: int = 4,
    cfg: float = 1.0,
    denoise: float = 1.0,
    builtin_style_strength: float | None = None,
    user_loras: list[dict[str, Any]] | None = None,
    suppress_builtin_style: bool = False,
    pose_reference_input_name: str | None = None,
    additional_reference_input_names: list[str] | None = None,
    noise_mask_input_name: str | None = None,
) -> dict[str, Any]:
    _sync_local_video_runtime(globals(), {"qwen_workflow"})
    effective_user_loras = normalize_user_lora_specs(
        user_loras,
        legacy_model_name=user_lora,
        legacy_strength=user_lora_strength,
        resolve_models=False,
    )
    active_user_loras = [
        item for item in effective_user_loras if float(item["strength"]) != 0.0
    ]
    effective_style_strength = (
        QWEN_STYLE_LORA_STRENGTH
        if builtin_style_strength is None
        else float(builtin_style_strength)
    )
    if not 0.0 <= effective_style_strength <= 2.0:
        raise ValueError("builtin_style_strength must be between 0.0 and 2.0")
    workflow: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": input_name}},
        "2": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": QWEN_MODEL, "weight_dtype": "default"},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": QWEN_CLIP,
                "type": "qwen_image",
                "device": "default",
            },
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "5": {"class_type": "FluxKontextImageScale", "inputs": {"image": link("1")}},
        "6": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": link("3"),
                "vae": link("4"),
                "image1": link("5"),
                "prompt": prompt,
            },
        },
        "7": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": link("3"),
                "vae": link("4"),
                "image1": link("5"),
                "prompt": negative_prompt,
            },
        },
        "8": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "conditioning": link("6"),
                "reference_latents_method": "index_timestep_zero",
            },
        },
        "9": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "conditioning": link("7"),
                "reference_latents_method": "index_timestep_zero",
            },
        },
        "10": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": link("5"), "vae": link("4")},
        },
        "11": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": link("2"), "shift": 3.1},
        },
        "12": {
            "class_type": "CFGNorm",
            "inputs": {"model": link("11"), "strength": 1.0},
        },
        "13": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": link("12"),
                "lora_name": QWEN_LORA,
                "strength_model": 1.0,
            },
        },
        "14": {
            "class_type": "KSampler",
            "inputs": {
                "model": link("13"),
                "positive": link("8"),
                "negative": link("9"),
                "latent_image": link("10"),
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": denoise,
            },
        },
        "15": {
            "class_type": "VAEDecode",
            "inputs": {"samples": link("14"), "vae": link("4")},
        },
        "16": {
            "class_type": "SaveImage",
            "inputs": {"images": link("15"), "filename_prefix": prefix},
        },
    }
    model_node = "13"
    apply_builtin_style = (
        bool(QWEN_STYLE_LORA)
        and not suppress_builtin_style
        and not (active_user_loras and user_lora_mode == "replace")
    )
    if apply_builtin_style:
        workflow["17"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": link("13"),
                "lora_name": QWEN_STYLE_LORA,
                "strength_model": effective_style_strength,
            },
        }
        model_node = "17"
    for lora_index, lora_spec in enumerate(active_user_loras, start=18):
        node_id = str(lora_index)
        workflow[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": link(model_node),
                "lora_name": lora_spec["model_name"],
                "strength_model": lora_spec["strength"],
            },
        }
        model_node = node_id
    workflow["14"]["inputs"]["model"] = link(model_node)
    if pose_reference_input_name and additional_reference_input_names:
        raise ValueError(
            "pose_reference_input_name and additional_reference_input_names "
            "cannot be used together"
        )
    reference_input_names = list(additional_reference_input_names or [])
    if pose_reference_input_name:
        reference_input_names = [pose_reference_input_name]
    if len(reference_input_names) > 2:
        raise ValueError("Qwen Image Edit Plus supports at most three images")
    for image_index, reference_input_name in enumerate(
        reference_input_names,
        start=2,
    ):
        if not str(reference_input_name).strip():
            raise ValueError("Qwen reference image names cannot be blank")
        next_node = max(int(node_id) for node_id in workflow) + 1
        reference_load_node = str(next_node)
        reference_scale_node = str(next_node + 1)
        workflow[reference_load_node] = {
            "class_type": "LoadImage",
            "inputs": {"image": reference_input_name},
        }
        workflow[reference_scale_node] = {
            "class_type": "FluxKontextImageScale",
            "inputs": {"image": link(reference_load_node)},
        }
        # Image 1 always remains the sampler latent. Canonical generation may add
        # the left and right identity views as Images 2 and 3. Production eating
        # and grooming both keep the approved canonical as Image 1. Advisory
        # grooming deliberately supplies no Image 2 or Image 3; its action is a
        # direct single-image edit instruction, never a video-frame or foreign-
        # pose reference image.
        input_key = f"image{image_index}"
        workflow["6"]["inputs"][input_key] = link(reference_scale_node)
        workflow["7"]["inputs"][input_key] = link(reference_scale_node)
    if noise_mask_input_name is not None:
        if not str(noise_mask_input_name).strip():
            raise ValueError("Qwen noise-mask image name cannot be blank")
        next_node = max(int(node_id) for node_id in workflow) + 1
        mask_load_node = str(next_node)
        masked_latent_node = str(next_node + 1)
        workflow[mask_load_node] = {
            "class_type": "LoadImage",
            "inputs": {"image": noise_mask_input_name},
        }
        workflow[masked_latent_node] = {
            "class_type": "SetLatentNoiseMask",
            "inputs": {
                "samples": link("10"),
                # ComfyUI LoadImage exposes MASK as 1 - alpha. The repair
                # mask is therefore transparent only where fresh noise and
                # a full-denoise local redraw are required.
                "mask": link(mask_load_node, 1),
            },
        }
        workflow["14"]["inputs"]["latent_image"] = link(masked_latent_node)
    return workflow
