/**
 * SmartScribe AI - Live Handwritten Digit Recognizer
 * SDU 바이브코딩 최주영 교수님 과제
 * 
 * 주요 기능:
 * 1. HTML5 Canvas 기반 마우스/터치 필기 스케치북
 * 2. 부드러운 브러시 곡선 및 한 획 되돌리기(Undo), 전체 지우기(Clear)
 * 3. 28x28 MNIST 표준 규격 전처리 (바운딩 박스 크기 보정 + 무게중심 정렬)
 * 4. TensorFlow.js GraphModel 기반 실시간 0~9 추론
 * 5. 직관적인 실시간 확률(0~100%) 막대 시각화
 */

// ==========================================
// 1. 전역 상태 및 DOM 요소 참조
// ==========================================
let model = null;
let isModelReady = false;
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let brushSize = 18;
const undoStack = [];
const MAX_UNDO = 25;
let debounceTimer = null;

// DOM 요소
const drawCanvas = document.getElementById('drawCanvas');
const drawCtx = drawCanvas.getContext('2d', { willReadFrequently: true });
const previewCanvas = document.getElementById('previewCanvas');
const previewCtx = previewCanvas.getContext('2d', { willReadFrequently: true });

const aiStatusBadge = document.getElementById('aiStatusBadge');
const aiStatusText = document.getElementById('aiStatusText');
const brushSizeInput = document.getElementById('brushSize');
const brushSizeDisplay = document.getElementById('brushSizeDisplay');
const canvasEmptyHint = document.getElementById('canvasEmptyHint');

const btnUndo = document.getElementById('btnUndo');
const btnClear = document.getElementById('btnClear');
const predictedDigit = document.getElementById('predictedDigit');
const confidenceBadge = document.getElementById('confidenceBadge');
const confidenceText = document.getElementById('confidenceText');
const inferenceTime = document.getElementById('inferenceTime');
const topRankSummary = document.getElementById('topRankSummary');
const probabilityList = document.getElementById('probabilityList');
const presetButtons = document.querySelectorAll('.preset-btn');

// ==========================================
// 2. 확률 막대 UI 초기 생성 (0~9)
// ==========================================
function initProbabilityList() {
  probabilityList.innerHTML = '';
  for (let i = 0; i <= 9; i++) {
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.id = `prob-row-${i}`;
    row.innerHTML = `
      <div class="prob-digit">${i}</div>
      <div class="prob-track">
        <div class="prob-bar" id="prob-bar-${i}"></div>
      </div>
      <div class="prob-percent" id="prob-percent-${i}">0.0%</div>
    `;
    probabilityList.appendChild(row);
  }
}

// ==========================================
// 3. 인공지능(AI) 모델 불러오기
// ==========================================
async function loadAIModel() {
  updateStatus('loading', 'AI 모델 불러오는 중...');
  
  // 브라우저 GPU 가속 백엔드 설정 (실패 시 CPU 백엔드로 안전하게 전환)
  try {
    if (tf.getBackend() !== 'webgl') {
      await tf.setBackend('webgl').catch(() => tf.setBackend('cpu'));
    }
    await tf.ready();
  } catch (backendErr) {
    console.warn('Backend setup notice:', backendErr);
  }

  const localModelUrl = './model/model.json';
  const cdnModelUrl = 'https://raw.githubusercontent.com/AnandkumarMall/MNIST-Handwritten-Digit-Recognizer/master/math-garden/TFJS/model.json';

  try {
    // 1차 시도: 로컬 모델 파일 로딩
    model = await tf.loadGraphModel(localModelUrl);
    console.log('✅ 로컬 모델 로드 완료');
  } catch (localErr) {
    console.warn('로컬 모델 로드 실패 (file:// 제약 등), CDN 원격 모델로 재시도합니다:', localErr);
    try {
      // 2차 시도: 원격 CDN 백업 모델 로딩
      model = await tf.loadGraphModel(cdnModelUrl);
      console.log('✅ CDN 원격 모델 로드 완료');
    } catch (cdnErr) {
      console.error('❌ 모든 모델 로딩 실패:', cdnErr);
      updateStatus('error', '모델 로딩 실패');
      return;
    }
  }

  // 워밍업 (첫 실행 딜레이 제거를 위한 더미 텐서 추론)
  try {
    tf.tidy(() => {
      const dummyInput = tf.zeros([1, 784], 'float32');
      executeModelInference(dummyInput);
    });
  } catch (warmupErr) {
    console.warn('Model warmup warning:', warmupErr);
  }

  isModelReady = true;
  updateStatus('ready', 'AI 실시간 판독 준비 완료');
}

function updateStatus(state, message) {
  aiStatusBadge.className = `status-badge ${state}`;
  aiStatusText.textContent = message;
}

// 모델 실행 래퍼 (GraphModel의 여러 출력 노드 중 확률 노드 추출)
function executeModelInference(inputTensor) {
  if (!model) return null;
  // out/Softmax 노드 직접 실행
  try {
    const outputs = model.execute(inputTensor, 'out/Softmax');
    return Array.isArray(outputs) ? outputs[0] : outputs;
  } catch (err) {
    // 기본 predict fallback
    return model.predict(inputTensor);
  }
}

// ==========================================
// 4. 스케치북 캔버스 그리기 로직
// ==========================================
function initCanvas() {
  // 캔버스를 순수 검정색(#000000)으로 초기화
  drawCtx.fillStyle = '#000000';
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  
  // 드로잉 스타일 설정
  drawCtx.strokeStyle = '#FFFFFF';
  drawCtx.lineCap = 'round';
  drawCtx.lineJoin = 'round';
  drawCtx.lineWidth = brushSize;

  // 마우스 이벤트 리스너
  drawCanvas.addEventListener('mousedown', startDrawing);
  window.addEventListener('mousemove', draw);
  window.addEventListener('mouseup', stopDrawing);

  // 모바일/태블릿 터치 이벤트 리스너
  drawCanvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = drawCanvas.getBoundingClientRect();
    startDrawing({
      clientX: touch.clientX,
      clientY: touch.clientY,
      target: drawCanvas
    });
  }, { passive: false });

  window.addEventListener('touchmove', (e) => {
    if (!isDrawing) return;
    const touch = e.touches[0];
    draw({
      clientX: touch.clientX,
      clientY: touch.clientY
    });
  }, { passive: false });

  window.addEventListener('touchend', stopDrawing);

  // 브러시 굵기 슬라이더 연동
  brushSizeInput.addEventListener('input', (e) => {
    brushSize = parseInt(e.target.value, 10);
    brushSizeDisplay.textContent = `${brushSize}px`;
    drawCtx.lineWidth = brushSize;
  });

  // 툴바 버튼 이벤트
  btnUndo.addEventListener('click', handleUndo);
  btnClear.addEventListener('click', handleClear);

  // 예시 프리셋 버튼 이벤트
  presetButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const digit = parseInt(btn.getAttribute('data-digit'), 10);
      drawPresetDigit(digit);
    });
  });
}

function getCanvasCoordinates(e) {
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY
  };
}

function saveCanvasState() {
  if (undoStack.length >= MAX_UNDO) {
    undoStack.shift();
  }
  undoStack.push(drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height));
}

function startDrawing(e) {
  if (e.target !== drawCanvas && !e.touches) return;
  isDrawing = true;
  saveCanvasState();
  
  const coords = getCanvasCoordinates(e);
  lastX = coords.x;
  lastY = coords.y;

  // 클릭만 했을 때도 점이 찍히도록
  drawCtx.beginPath();
  drawCtx.arc(lastX, lastY, brushSize / 2, 0, Math.PI * 2);
  drawCtx.fillStyle = '#FFFFFF';
  drawCtx.fill();

  canvasEmptyHint.classList.add('hidden');
  scheduleRealtimeRecognition();
}

function draw(e) {
  if (!isDrawing) return;
  const coords = getCanvasCoordinates(e);

  drawCtx.beginPath();
  drawCtx.moveTo(lastX, lastY);
  drawCtx.lineTo(coords.x, coords.y);
  drawCtx.stroke();

  lastX = coords.x;
  lastY = coords.y;

  scheduleRealtimeRecognition();
}

function stopDrawing() {
  if (!isDrawing) return;
  isDrawing = false;
  // 손을 떼는 순간 즉시 최종 판독 실행
  performRecognition();
}

function handleUndo() {
  if (undoStack.length > 0) {
    const prevState = undoStack.pop();
    drawCtx.putImageData(prevState, 0, 0);
    checkCanvasEmpty();
    performRecognition();
  }
}

function handleClear() {
  saveCanvasState();
  drawCtx.fillStyle = '#000000';
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  canvasEmptyHint.classList.remove('hidden');
  resetRecognitionDisplay();
}

function checkCanvasEmpty() {
  const imgData = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
  let hasContent = false;
  for (let i = 0; i < imgData.data.length; i += 4) {
    if (imgData.data[i] > 20) { // R 채널 검사
      hasContent = true;
      break;
    }
  }
  if (!hasContent) {
    canvasEmptyHint.classList.remove('hidden');
  } else {
    canvasEmptyHint.classList.add('hidden');
  }
  return !hasContent;
}

// 실시간 디바운스 추론 예약 (100ms 간격으로 부드럽게 갱신)
function scheduleRealtimeRecognition() {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    performRecognition();
  }, 100);
}

// ==========================================
// 5. MNIST 표준 이미지 전처리 알고리즘
// ==========================================
/**
 * 인공지능이 가장 잘 알아볼 수 있도록
 * 캔버스의 그림을 28x28 크기로 정규화하고 무게중심을 맞춥니다.
 */
function preprocessCanvas() {
  const width = drawCanvas.width;
  const height = drawCanvas.height;
  const imgData = drawCtx.getImageData(0, 0, width, height);
  const data = imgData.data;

  // 1) 그려진 글씨의 바운딩 박스(최소/최대 사각 영역) 탐색
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  let totalMass = 0;
  let weightedX = 0;
  let weightedY = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const intensity = data[idx]; // R 채널 (0~255)
      if (intensity > 25) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;

        totalMass += intensity;
        weightedX += x * intensity;
        weightedY += y * intensity;
      }
    }
  }

  // 픽셀이 거의 없으면 빈 화면으로 판정
  if (totalMass < 200 || maxX < minX || maxY < minY) {
    return null;
  }

  // 2) 바운딩 박스 영역 추출
  const boxWidth = maxX - minX + 1;
  const boxHeight = maxY - minY + 1;

  // 임시 캔버스에 바운딩 박스 영역 복사
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = boxWidth;
  tempCanvas.height = boxHeight;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(
    drawCanvas,
    minX, minY, boxWidth, boxHeight,
    0, 0, boxWidth, boxHeight
  );

  // 3) 가로/세로 비율을 유지하면서 최대 20픽셀 안에 들어가도록 축소
  const targetSize = 20;
  let scaledW, scaledH;
  if (boxWidth > boxHeight) {
    scaledW = targetSize;
    scaledH = Math.max(1, Math.round((boxHeight / boxWidth) * targetSize));
  } else {
    scaledH = targetSize;
    scaledW = Math.max(1, Math.round((boxWidth / boxHeight) * targetSize));
  }

  const scaledCanvas = document.createElement('canvas');
  scaledCanvas.width = scaledW;
  scaledCanvas.height = scaledH;
  const scaledCtx = scaledCanvas.getContext('2d');
  scaledCtx.imageSmoothingEnabled = true;
  scaledCtx.imageSmoothingQuality = 'high';
  scaledCtx.drawImage(tempCanvas, 0, 0, scaledW, scaledH);

  // 4) 임시 28x28 도화지에 1차 배치
  const temp28 = document.createElement('canvas');
  temp28.width = 28;
  temp28.height = 28;
  const temp28Ctx = temp28.getContext('2d');
  temp28Ctx.fillStyle = '#000000';
  temp28Ctx.fillRect(0, 0, 28, 28);

  const initialX = Math.round((28 - scaledW) / 2);
  const initialY = Math.round((28 - scaledH) / 2);
  temp28Ctx.drawImage(scaledCanvas, initialX, initialY);

  // 5) MNIST 표준 규격: 28x28 이미지의 픽셀 무게중심(Center of Mass) 계산
  const temp28Data = temp28Ctx.getImageData(0, 0, 28, 28).data;
  let mass28 = 0;
  let cx28 = 0;
  let cy28 = 0;

  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const val = temp28Data[(y * 28 + x) * 4];
      if (val > 10) {
        mass28 += val;
        cx28 += x * val;
        cy28 += y * val;
      }
    }
  }

  // 무게중심이 정확히 (14, 14)에 오도록 미세 이동(Shift) 계산
  let shiftX = 0;
  let shiftY = 0;
  if (mass28 > 0) {
    cx28 /= mass28;
    cy28 /= mass28;
    shiftX = Math.round(14.0 - cx28);
    shiftY = Math.round(14.0 - cy28);
    
    // 외곽으로 너무 벗어나지 않도록 클램핑 (-4 ~ +4)
    shiftX = Math.max(-4, Math.min(4, shiftX));
    shiftY = Math.max(-4, Math.min(4, shiftY));
  }

  // 6) 최종 previewCanvas(28x28)에 무게중심 정렬된 이미지 전송
  previewCtx.fillStyle = '#000000';
  previewCtx.fillRect(0, 0, 28, 28);
  previewCtx.drawImage(temp28, shiftX, shiftY);

  // 7) 28x28 픽셀 데이터를 추출하여 [0, 1] 범위의 Float32 배열(길이 784)로 변환
  const finalImgData = previewCtx.getImageData(0, 0, 28, 28);
  const finalPixels = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    finalPixels[i] = finalImgData.data[i * 4] / 255.0; // 0.0 ~ 1.0 정규화
  }

  return finalPixels;
}

// ==========================================
// 6. 실시간 판독 및 UI 갱신
// ==========================================
async function performRecognition() {
  if (!isModelReady || !model) return;

  const startTime = performance.now();
  const inputPixels = preprocessCanvas();

  if (!inputPixels) {
    resetRecognitionDisplay();
    return;
  }

  let probabilities = null;
  tf.tidy(() => {
    const inputTensor = tf.tensor2d(inputPixels, [1, 784], 'float32');
    const outputTensor = executeModelInference(inputTensor);
    if (outputTensor) {
      probabilities = Array.from(outputTensor.dataSync());
    }
  });

  const duration = Math.round(performance.now() - startTime);
  inferenceTime.textContent = `반응속도: ${duration} ms`;

  if (probabilities && probabilities.length === 10) {
    updatePredictionUI(probabilities);
  }
}

function updatePredictionUI(probabilities) {
  // 순위 정렬
  const ranked = probabilities
    .map((prob, digit) => ({ digit, prob }))
    .sort((a, b) => b.prob - a.prob);

  const top1 = ranked[0];
  const top2 = ranked[1];
  const top1Percent = (top1.prob * 100).toFixed(1);
  const top2Percent = (top2.prob * 100).toFixed(1);

  // 1등 숫자 표시
  predictedDigit.textContent = top1.digit;
  predictedDigit.classList.add('pop-anim');
  setTimeout(() => predictedDigit.classList.remove('pop-anim'), 180);

  // 신뢰도 뱃지 스타일링
  confidenceBadge.className = 'confidence-badge';
  if (top1.prob >= 0.80) {
    confidenceBadge.classList.add('high');
    confidenceText.textContent = `신뢰도 높음 · ${top1Percent}%`;
  } else if (top1.prob >= 0.50) {
    confidenceBadge.classList.add('medium');
    confidenceText.textContent = `신뢰도 보통 · ${top1Percent}%`;
  } else {
    confidenceBadge.classList.add('low');
    confidenceText.textContent = `신뢰도 낮음 · ${top1Percent}%`;
  }

  topRankSummary.textContent = `1위: ${top1.digit} (${top1Percent}%) / 2위: ${top2.digit} (${top2Percent}%)`;

  // 각 0~9 행 막대그래프 업데이트
  for (let i = 0; i <= 9; i++) {
    const row = document.getElementById(`prob-row-${i}`);
    const bar = document.getElementById(`prob-bar-${i}`);
    const percentLabel = document.getElementById(`prob-percent-${i}`);

    const percentVal = (probabilities[i] * 100).toFixed(1);
    bar.style.width = `${percentVal}%`;
    percentLabel.textContent = `${percentVal}%`;

    row.className = 'prob-row';
    if (i === top1.digit) {
      row.classList.add('top-rank');
    } else if (i === top2.digit && top2.prob > 0.05) {
      row.classList.add('runner-up');
    }
  }
}

function resetRecognitionDisplay() {
  predictedDigit.textContent = '-';
  confidenceBadge.className = 'confidence-badge';
  confidenceText.textContent = '신뢰도 대기 중';
  inferenceTime.textContent = '반응속도: - ms';
  topRankSummary.textContent = '';

  // 미리보기 캔버스도 검정색으로 클리어
  previewCtx.fillStyle = '#000000';
  previewCtx.fillRect(0, 0, 28, 28);

  for (let i = 0; i <= 9; i++) {
    const row = document.getElementById(`prob-row-${i}`);
    const bar = document.getElementById(`prob-bar-${i}`);
    const percentLabel = document.getElementById(`prob-percent-${i}`);
    if (bar && percentLabel && row) {
      bar.style.width = '0%';
      percentLabel.textContent = '0.0%';
      row.className = 'prob-row';
    }
  }
}

// ==========================================
// 7. 빠른 시연을 위한 숫자 프리셋 그리기
// ==========================================
function drawPresetDigit(digit) {
  handleClear();
  canvasEmptyHint.classList.add('hidden');

  drawCtx.strokeStyle = '#FFFFFF';
  drawCtx.lineWidth = brushSize;
  drawCtx.lineCap = 'round';
  drawCtx.lineJoin = 'round';

  const cx = 140;
  const cy = 140;

  drawCtx.beginPath();
  switch (digit) {
    case 0:
      drawCtx.ellipse(cx, cy, 45, 65, 0, 0, Math.PI * 2);
      break;
    case 1:
      drawCtx.moveTo(cx - 15, cy - 50);
      drawCtx.lineTo(cx, cy - 65);
      drawCtx.lineTo(cx, cy + 65);
      break;
    case 2:
      drawCtx.arc(cx, cy - 30, 40, Math.PI, 0, false);
      drawCtx.lineTo(cx - 40, cy + 65);
      drawCtx.lineTo(cx + 45, cy + 65);
      break;
    case 3:
      drawCtx.arc(cx, cy - 30, 35, -Math.PI * 0.7, Math.PI * 0.5, false);
      drawCtx.arc(cx, cy + 30, 38, -Math.PI * 0.5, Math.PI * 0.8, false);
      break;
    case 4:
      drawCtx.moveTo(cx + 20, cy + 60);
      drawCtx.lineTo(cx + 20, cy - 60);
      drawCtx.moveTo(cx - 35, cy - 50);
      drawCtx.lineTo(cx - 35, cy + 15);
      drawCtx.lineTo(cx + 40, cy + 15);
      break;
    case 5:
      drawCtx.moveTo(cx + 30, cy - 60);
      drawCtx.lineTo(cx - 25, cy - 60);
      drawCtx.lineTo(cx - 28, cy - 8);
      drawCtx.arc(cx - 2, cy + 18, 42, -Math.PI * 0.5, Math.PI * 0.75, false);
      break;
    case 6:
      drawCtx.moveTo(cx + 15, cy - 60);
      drawCtx.quadraticCurveTo(cx - 42, cy - 30, cx - 42, cy + 20);
      drawCtx.arc(cx, cy + 20, 42, Math.PI, Math.PI * 3, false);
      break;
    case 7:
      drawCtx.moveTo(cx - 35, cy - 60);
      drawCtx.lineTo(cx + 35, cy - 60);
      drawCtx.lineTo(cx - 5, cy + 60);
      drawCtx.moveTo(cx - 15, cy - 2);
      drawCtx.lineTo(cx + 15, cy - 2);
      break;
    case 8:
      drawCtx.arc(cx, cy - 30, 30, 0, Math.PI * 2);
      drawCtx.arc(cx, cy + 30, 35, 0, Math.PI * 2);
      break;
    case 9:
      drawCtx.arc(cx, cy - 22, 38, 0, Math.PI * 2, false);
      drawCtx.moveTo(cx + 38, cy - 22);
      drawCtx.quadraticCurveTo(cx + 38, cy + 45, cx - 15, cy + 60);
      break;
  }
  drawCtx.stroke();
  performRecognition();
}

// ==========================================
// 8. 초기화 실행
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
  initProbabilityList();
  initCanvas();
  loadAIModel();
});
