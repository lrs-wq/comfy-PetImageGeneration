const canvas = document.querySelector("#avatarCanvas");
const ctx = canvas.getContext("2d");

const ui = {
  petUpload: document.querySelector("#petUpload"),
  samplePetButton: document.querySelector("#samplePetButton"),
  backgroundSwatches: document.querySelector("#backgroundSwatches"),
  adjustScale: document.querySelector("#adjustScale"),
  adjustScaleValue: document.querySelector("#adjustScaleValue"),
  adjustX: document.querySelector("#adjustX"),
  adjustXValue: document.querySelector("#adjustXValue"),
  adjustY: document.querySelector("#adjustY"),
  adjustYValue: document.querySelector("#adjustYValue"),
  adjustRotate: document.querySelector("#adjustRotate"),
  adjustRotateValue: document.querySelector("#adjustRotateValue"),
  randomButton: document.querySelector("#randomButton"),
  clearButton: document.querySelector("#clearButton"),
  refineButton: document.querySelector("#refineButton"),
  exportButton: document.querySelector("#exportButton"),
  refineStatus: document.querySelector("#refineStatus"),
  refinedResult: document.querySelector("#refinedResult"),
  refinedImage: document.querySelector("#refinedImage"),
  openRefinedLink: document.querySelector("#openRefinedLink"),
  canvasFrame: document.querySelector("#canvasFrame"),
  outfitStack: document.querySelector("#outfitStack"),
  slotTabs: document.querySelector("#slotTabs"),
  wardrobeGrid: document.querySelector("#wardrobeGrid")
};

const backgrounds = [
  { id: "transparent", label: "透明", className: "swatch-transparent" },
  { id: "studio", label: "影棚", fill: ["#f9fafc", "#dfe5ef"] },
  { id: "mint", label: "薄荷", fill: ["#dff7ee", "#f7fbff"] },
  { id: "peach", label: "蜜桃", fill: ["#ffe1d2", "#fff7e6"] }
];

const state = {
  avatarManifest: null,
  wardrobeManifest: null,
  avatar: null,
  activeSlot: "top",
  selected: {
    back: null,
    top: "red_hoodie",
    collar: "bell_collar",
    eyes: null,
    hat: "beanie"
  },
  adjustments: {},
  background: "transparent",
  images: new Map(),
  customPetImage: null,
  customPetUrl: null
};

function distance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function imageFromSrc(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Image failed to load: ${src}`));
    image.src = src;
  });
}

async function loadImage(src) {
  if (!src) return null;
  if (!state.images.has(src)) {
    state.images.set(src, await imageFromSrc(src));
  }
  return state.images.get(src);
}

async function loadData() {
  const [avatarManifest, wardrobeManifest] = await Promise.all([
    fetch("data/avatar_manifest.json").then((response) => response.json()),
    fetch("data/wardrobe_manifest.json").then((response) => response.json())
  ]);
  state.avatarManifest = avatarManifest;
  state.wardrobeManifest = wardrobeManifest;
  state.avatar = avatarManifest.avatars[0];

  const sources = new Set([state.avatar.image, state.avatar.overlays.head, state.avatar.overlays.frontPaws]);
  wardrobeManifest.items.forEach((item) => sources.add(item.asset));
  await Promise.all([...sources].map((src) => loadImage(src)));
}

function currentItem() {
  const itemId = state.selected[state.activeSlot];
  if (!itemId) return null;
  return state.wardrobeManifest.items.find((item) => item.id === itemId) || null;
}

function selectedOutfitItems() {
  return Object.values(state.selected)
    .filter(Boolean)
    .map((id) => state.wardrobeManifest.items.find((item) => item.id === id))
    .filter(Boolean);
}

function itemAdjustment(itemId) {
  if (!state.adjustments[itemId]) {
    state.adjustments[itemId] = { scale: 1, x: 0, y: 0, rotate: 0 };
  }
  return state.adjustments[itemId];
}

function measureFor(item) {
  const measurement = state.avatar.measurements[item.scaleBy];
  if (!measurement) return item.referenceDistance || 1;
  return distance(state.avatar.anchors[measurement[0]], state.avatar.anchors[measurement[1]]);
}

function itemScale(item) {
  const adjustment = itemAdjustment(item.id);
  return (measureFor(item) / item.referenceDistance) * item.scale * adjustment.scale;
}

function drawBackground() {
  const background = backgrounds.find((entry) => entry.id === state.background);
  if (!background || background.id === "transparent") return;

  const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
  gradient.addColorStop(0, background.fill[0]);
  gradient.addColorStop(1, background.fill[1]);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.save();
  ctx.globalAlpha = 0.18;
  ctx.fillStyle = "#2d3441";
  ctx.beginPath();
  ctx.ellipse(canvas.width / 2, canvas.height - 80, canvas.width * 0.3, 30, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawCanonicalPet(image) {
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
}

function drawCustomPet(image) {
  const maxWidth = canvas.width * 0.92;
  const maxHeight = canvas.height * 0.92;
  const scale = Math.min(maxWidth / image.naturalWidth, maxHeight / image.naturalHeight);
  const width = image.naturalWidth * scale;
  const height = image.naturalHeight * scale;
  const x = (canvas.width - width) / 2;
  const y = canvas.height - height - 12;
  ctx.drawImage(image, x, y, width, height);
}

async function drawItem(slotId) {
  const itemId = state.selected[slotId];
  if (!itemId) return;

  const item = state.wardrobeManifest.items.find((entry) => entry.id === itemId);
  if (!item) return;

  const image = await loadImage(item.asset);
  const target = state.avatar.anchors[item.targetAnchor];
  const adjustment = itemAdjustment(item.id);
  const scale = itemScale(item);
  const x = target[0] + item.offset[0] + adjustment.x;
  const y = target[1] + item.offset[1] + adjustment.y;
  const rotate = (adjustment.rotate * Math.PI) / 180;

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rotate);
  ctx.drawImage(
    image,
    -item.assetAnchor[0] * scale,
    -item.assetAnchor[1] * scale,
    image.naturalWidth * scale,
    image.naturalHeight * scale
  );
  ctx.restore();
}

async function render() {
  const width = state.avatar.canvas.width;
  const height = state.avatar.canvas.height;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawBackground();

  await drawItem("back");

  if (state.customPetImage) {
    drawCustomPet(state.customPetImage);
  } else {
    drawCanonicalPet(await loadImage(state.avatar.image));
  }

  await drawItem("top");

  if (!state.customPetImage) {
    drawCanonicalPet(await loadImage(state.avatar.overlays.head));
    drawCanonicalPet(await loadImage(state.avatar.overlays.frontPaws));
  }

  await drawItem("collar");
  await drawItem("eyes");
  await drawItem("hat");

  renderOutfitStack();
}

function renderBackgroundSwatches() {
  ui.backgroundSwatches.innerHTML = "";
  backgrounds.forEach((background) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `swatch ${background.className || ""}`;
    button.title = background.label;
    button.setAttribute("aria-label", background.label);
    if (background.fill) {
      button.style.background = `linear-gradient(135deg, ${background.fill[0]}, ${background.fill[1]})`;
    }
    if (background.id === state.background) button.classList.add("is-active");
    button.addEventListener("click", () => {
      state.background = background.id;
      renderBackgroundSwatches();
      render();
    });
    ui.backgroundSwatches.append(button);
  });
}

function renderSlotTabs() {
  ui.slotTabs.innerHTML = "";
  state.wardrobeManifest.slots.forEach((slot) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "slot-tab";
    button.textContent = slot.label;
    if (slot.id === state.activeSlot) button.classList.add("is-active");
    button.addEventListener("click", () => {
      state.activeSlot = slot.id;
      renderSlotTabs();
      renderWardrobeGrid();
      syncAdjustmentControls();
    });
    ui.slotTabs.append(button);
  });
}

function emptyCard(slot) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = "item-card";
  if (!state.selected[slot.id]) card.classList.add("is-selected");

  const thumb = document.createElement("div");
  thumb.className = "item-thumb";
  const empty = document.createElement("div");
  empty.className = "empty-thumb";
  empty.textContent = "×";
  thumb.append(empty);

  const label = document.createElement("div");
  label.className = "item-name";
  label.textContent = "不穿";

  card.append(thumb, label);
  card.addEventListener("click", () => {
    state.selected[slot.id] = null;
    renderWardrobeGrid();
    syncAdjustmentControls();
    render();
  });
  return card;
}

function itemCard(item) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = "item-card";
  if (state.selected[item.slot] === item.id) card.classList.add("is-selected");

  const thumb = document.createElement("div");
  thumb.className = "item-thumb";
  const image = document.createElement("img");
  image.alt = item.name;
  image.src = item.asset;
  thumb.append(image);

  const label = document.createElement("div");
  label.className = "item-name";
  label.textContent = item.name;

  card.append(thumb, label);
  card.addEventListener("click", () => {
    state.selected[item.slot] = item.id;
    renderWardrobeGrid();
    syncAdjustmentControls();
    render();
  });
  return card;
}

function renderWardrobeGrid() {
  const slot = state.wardrobeManifest.slots.find((entry) => entry.id === state.activeSlot);
  const items = state.wardrobeManifest.items.filter((item) => item.slot === state.activeSlot);
  ui.wardrobeGrid.innerHTML = "";
  ui.wardrobeGrid.append(emptyCard(slot));
  items.forEach((item) => ui.wardrobeGrid.append(itemCard(item)));
}

function renderOutfitStack() {
  ui.outfitStack.innerHTML = "";
  const selectedItems = selectedOutfitItems();

  if (selectedItems.length === 0) {
    const chip = document.createElement("span");
    chip.className = "outfit-chip";
    chip.textContent = "无装扮";
    ui.outfitStack.append(chip);
    return;
  }

  selectedItems.forEach((item) => {
    const chip = document.createElement("span");
    chip.className = "outfit-chip";
    chip.textContent = item.name;
    ui.outfitStack.append(chip);
  });
}

function setRefineStatus(message, mode = "") {
  ui.refineStatus.textContent = message;
  ui.refineStatus.className = `status-text ${mode}`.trim();
}

function syncAdjustmentControls() {
  const item = currentItem();
  const disabled = !item;
  [ui.adjustScale, ui.adjustX, ui.adjustY, ui.adjustRotate].forEach((input) => {
    input.disabled = disabled;
  });

  if (!item) {
    ui.adjustScale.value = 100;
    ui.adjustX.value = 0;
    ui.adjustY.value = 0;
    ui.adjustRotate.value = 0;
  } else {
    const adjustment = itemAdjustment(item.id);
    ui.adjustScale.value = Math.round(adjustment.scale * 100);
    ui.adjustX.value = Math.round(adjustment.x);
    ui.adjustY.value = Math.round(adjustment.y);
    ui.adjustRotate.value = Math.round(adjustment.rotate);
  }

  ui.adjustScaleValue.textContent = `${ui.adjustScale.value}%`;
  ui.adjustXValue.textContent = ui.adjustX.value;
  ui.adjustYValue.textContent = ui.adjustY.value;
  ui.adjustRotateValue.textContent = `${ui.adjustRotate.value}°`;
}

function updateAdjustmentFromControls() {
  const item = currentItem();
  if (!item) return;
  const adjustment = itemAdjustment(item.id);
  adjustment.scale = Number(ui.adjustScale.value) / 100;
  adjustment.x = Number(ui.adjustX.value);
  adjustment.y = Number(ui.adjustY.value);
  adjustment.rotate = Number(ui.adjustRotate.value);
  syncAdjustmentControls();
  render();
}

function chooseRandom() {
  state.wardrobeManifest.slots.forEach((slot) => {
    const items = state.wardrobeManifest.items.filter((item) => item.slot === slot.id);
    const includeNone = slot.id === "back" || slot.id === "eyes";
    const choices = includeNone ? [null, ...items.map((item) => item.id)] : items.map((item) => item.id);
    state.selected[slot.id] = choices[Math.floor(Math.random() * choices.length)] || null;
  });
  renderSlotTabs();
  renderWardrobeGrid();
  syncAdjustmentControls();
  render();
}

function clearOutfit() {
  Object.keys(state.selected).forEach((slotId) => {
    state.selected[slotId] = null;
  });
  renderSlotTabs();
  renderWardrobeGrid();
  syncAdjustmentControls();
  render();
}

async function handleUpload(event) {
  const [file] = event.target.files;
  if (!file) return;
  if (state.customPetUrl) URL.revokeObjectURL(state.customPetUrl);
  state.customPetUrl = URL.createObjectURL(file);
  state.customPetImage = await imageFromSrc(state.customPetUrl);
  render();
}

function restoreSamplePet() {
  if (state.customPetUrl) URL.revokeObjectURL(state.customPetUrl);
  state.customPetUrl = null;
  state.customPetImage = null;
  ui.petUpload.value = "";
  render();
}

function exportPng() {
  render().then(() => {
    const link = document.createElement("a");
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    link.download = `pet-dressup-${timestamp}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  });
}

async function refineOutfit() {
  ui.refineButton.disabled = true;
  setRefineStatus("正在调用模型融合，通常需要几十秒...", "is-busy");
  ui.openRefinedLink.classList.remove("is-visible");

  try {
    await render();
    const outfitNames = selectedOutfitItems().map((item) => item.name);
    const response = await fetch("api/refine-outfit", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        imageDataUrl: canvas.toDataURL("image/png"),
        outfit: outfitNames,
        background: state.background
      })
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }

    const resultUrl = `${payload.refinedUrl}?t=${Date.now()}`;
    ui.refinedImage.src = resultUrl;
    ui.refinedResult.hidden = false;
    ui.openRefinedLink.href = payload.refinedUrl;
    ui.openRefinedLink.classList.add("is-visible");
    setRefineStatus(`已完成：${payload.jobId}`, "is-done");
  } catch (error) {
    setRefineStatus(`融合失败：${error.message}`, "is-error");
  } finally {
    ui.refineButton.disabled = false;
  }
}

function bindEvents() {
  ui.petUpload.addEventListener("change", handleUpload);
  ui.samplePetButton.addEventListener("click", restoreSamplePet);
  ui.randomButton.addEventListener("click", chooseRandom);
  ui.clearButton.addEventListener("click", clearOutfit);
  ui.exportButton.addEventListener("click", exportPng);
  ui.refineButton.addEventListener("click", refineOutfit);
  [ui.adjustScale, ui.adjustX, ui.adjustY, ui.adjustRotate].forEach((input) => {
    input.addEventListener("input", updateAdjustmentFromControls);
  });
}

async function init() {
  bindEvents();
  await loadData();
  renderBackgroundSwatches();
  renderSlotTabs();
  renderWardrobeGrid();
  syncAdjustmentControls();
  await render();
}

init().catch((error) => {
  console.error(error);
  ui.outfitStack.textContent = error.message;
});
