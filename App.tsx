
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { FilesetResolver, HandLandmarker, NormalizedLandmark } from '@mediapipe/tasks-vision';

// --- Constants ---
const SCATTER_CENTER_Y = 0.2;
const SCATTER_CONFIG = {
    CORE_RATIO: 0.8,
    CORE_RADIUS: 3.2,
    OUTER_RADIUS: 16.0,
};

// --- Photo Physics Constants ---
const PHOTO_DAMP = 0.08;
const ROT_DAMP   = 0.08;
const AUTO_ROT_SPEED = 0.05;

const WISH_LINES = [
  "Merry Christmas 🎄",
  "Warm Cheers from Hoenergy 🥂",
  "Happy New Year 🎉",
  "Stay Energized ⚡",
  "Good Luck 🍀"
];

// --- Shaders ---
const bodyVertexShader = `
  uniform float uMorph;
  uniform float uTime;
  uniform float uSizeScale;
  attribute vec3 aScatterPosition;
  attribute vec3 aColor;
  attribute float aSpark; 
  attribute float aSeed;
  attribute float aHalo; 
  
  varying vec3 vColor;
  varying float vHeight;
  varying float vSpark;
  varying float vSeed;

  float easeInOutCubic(float x) {
    return x < 0.5 ? 4.0 * x * x * x : 1.0 - pow(-2.0 * x + 2.0, 3.0) / 2.0;
  }

  void main() {
    float easedMorph = easeInOutCubic(uMorph);
    vec3 mixedPos = mix(position, aScatterPosition, easedMorph);
    
    if (uMorph < 0.99) {
       float breathe = sin(uTime * 1.2 + mixedPos.y * 0.5) * 0.02 * (1.0 - easedMorph);
       mixedPos.x += breathe;
       mixedPos.z += breathe;
    }
    
    if (uMorph > 0.01) {
       vec3 center = vec3(0.0, ${SCATTER_CENTER_Y.toFixed(1)}, 0.0);
       if (aHalo > 0.5) {
           float noiseX = sin(uTime * 0.4 + aSeed * 12.0);
           float noiseY = cos(uTime * 0.3 + aSeed * 25.0);
           float noiseZ = sin(uTime * 0.2 + mixedPos.x);
           vec3 drift = vec3(noiseX, noiseY * 0.5, noiseZ) * 0.4;
           mixedPos += drift * easedMorph;
       } else {
           vec3 dir = normalize(mixedPos - center);
           if (length(mixedPos - center) < 0.001) dir = vec3(0.0, 1.0, 0.0);
           float pulse = sin(uTime * 1.5 + aSeed * 10.0 + mixedPos.y);
           mixedPos += dir * 0.05 * pulse * easedMorph;
       }
    }
    
    vec4 mvPosition = modelViewMatrix * vec4(mixedPos, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    
    float h = clamp((mixedPos.y + 5.0) / 12.0, 0.0, 1.0);
    vHeight = h;
    vSpark = aSpark;
    vSeed = aSeed;
    
    float sizeGradient = mix(1.2, 0.6, h); 
    float outerScale = (aHalo > 0.5) ? 0.7 : 1.0; 
    
    gl_PointSize = uSizeScale * (15.0 / -mvPosition.z) * sizeGradient * outerScale;
    vColor = aColor * mix(1.0, 0.5, h);
  }
`;

const bodyFragmentShader = `
  varying vec3 vColor;
  varying float vHeight;
  varying float vSpark;
  varying float vSeed;
  uniform float uOpacity;
  uniform float uTime;
  void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard; 
    float baseAlpha = uOpacity * mix(0.7, 0.35, vHeight);
    float brightnessMult = mix(1.0, 2.2, vSpark);
    if (vSpark > 0.5) {
        float twinkle = 0.85 + 0.15 * sin(uTime * 2.0 + vSeed * 10.0);
        brightnessMult *= twinkle;
    }
    vec3 finalColor = vColor * brightnessMult;
    float finalAlpha = baseAlpha * mix(1.0, 1.4, vSpark);
    finalColor = min(finalColor, vec3(0.95));
    gl_FragColor = vec4(finalColor, finalAlpha);
  }
`;
const mixVertexShader = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
  }
`;
const mixFragmentShader = `
  uniform sampler2D baseTexture;
  uniform sampler2D bloomTexture;
  varying vec2 vUv;
  void main() {
    vec4 base = texture2D( baseTexture, vUv );
    vec4 bloom = texture2D( bloomTexture, vUv );
    gl_FragColor = base + bloom; 
  }
`;

const getIsotropicScatterPos = (isCoreOverride?: boolean) => {
    const isCore = isCoreOverride !== undefined ? isCoreOverride : Math.random() < SCATTER_CONFIG.CORE_RATIO;
    let pos = new THREE.Vector3();
    const center = new THREE.Vector3(0, SCATTER_CENTER_Y, 0);
    const u = Math.random();
    const v = Math.random();
    const theta = 2 * Math.PI * u;
    const phi = Math.acos(2 * v - 1);
    const dir = new THREE.Vector3(Math.sin(phi) * Math.cos(theta), Math.sin(phi) * Math.sin(theta), Math.cos(phi));

    if (isCore) {
        const r = SCATTER_CONFIG.CORE_RADIUS * Math.cbrt(Math.random());
        pos.copy(dir).multiplyScalar(r).add(center);
    } else {
        const R_min = SCATTER_CONFIG.CORE_RADIUS * 1.05;
        const R_max = SCATTER_CONFIG.OUTER_RADIUS;
        const r = Math.cbrt(Math.random() * (Math.pow(R_max, 3) - Math.pow(R_min, 3)) + Math.pow(R_min, 3));
        pos.copy(dir).multiplyScalar(r).add(center);
        if (Math.random() < 0.3) {
             const seed = Math.random() * 100;
             pos.x += 0.15 * Math.sin(pos.y * 1.7 + seed);
             pos.z += 0.15 * Math.cos(pos.x * 1.7 + seed);
        }
    }
    return { pos, isHalo: !isCore };
};

class MorphingInstances {
  mesh: THREE.InstancedMesh;
  treePositions: THREE.Vector3[];
  scatterPositions: THREE.Vector3[];
  haloFlags: boolean[];
  dummy: THREE.Object3D;

  constructor(mesh: THREE.InstancedMesh, count: number) {
    this.mesh = mesh;
    this.treePositions = [];
    this.scatterPositions = [];
    this.haloFlags = [];
    this.dummy = new THREE.Object3D();
  }

  update(morphVal: number, time: number) {
    const eased = morphVal < 0.5 ? 4.0 * morphVal * morphVal * morphVal : 1.0 - Math.pow(-2.0 * morphVal + 2.0, 3.0) / 2.0;
    const center = new THREE.Vector3(0, SCATTER_CENTER_Y, 0);
    for (let i = 0; i < this.mesh.count; i++) {
      const t = this.treePositions[i];
      const s = this.scatterPositions[i];
      const isOuter = this.haloFlags[i];
      this.dummy.position.lerpVectors(t, s, eased);
      if (morphVal > 0.01) {
          if (isOuter) {
               const noiseX = Math.sin(time * 0.4 + i * 0.1);
               const noiseY = Math.cos(time * 0.3 + i * 0.2);
               this.dummy.position.x += noiseX * 0.02 * eased;
               this.dummy.position.y += noiseY * 0.02 * eased;
          } else {
               const dir = new THREE.Vector3().subVectors(s, center).normalize();
               const noise = Math.sin(time * 2.0 + i * 0.1);
               this.dummy.position.addScaledVector(dir, 0.05 * eased * noise);
          }
      }
      this.dummy.rotation.x = i * 0.1;
      this.dummy.rotation.y = i * 0.2;
      this.dummy.updateMatrix();
      this.mesh.setMatrixAt(i, this.dummy.matrix);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}

type PerformanceMode = 'High' | 'Medium' | 'Low';
type HandStatus = 'OFF' | 'LOADING' | 'ON' | 'ERROR';
type GestureType = 'FIST' | 'OPEN_PALM' | 'V_SIGN' | 'INDEX_UP' | 'UNKNOWN';

interface PhotoData {
  mesh: THREE.Mesh;
  treePos: THREE.Vector3;
  scatterPos: THREE.Vector3;
  idx: number;
  baseScale: THREE.Vector3;
}

const App: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const handCanvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [targetMorph, setTargetMorph] = useState<number>(0);
  const [perfMode, setPerfMode] = useState<PerformanceMode>('High');
  const [handStatus, setHandStatus] = useState<HandStatus>('OFF');
  const [uiVisible, setUiVisible] = useState(true);
  const [wishIndex, setWishIndex] = useState(0);
  const [debugGesture, setDebugGesture] = useState<string>('--');
  const [isHandRotating, setIsHandRotating] = useState(false);

  const sceneRef = useRef<THREE.Scene | null>(null);
  const sceneRootRef = useRef<THREE.Group | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const bloomPassRef = useRef<UnrealBloomPass | null>(null);
  const bodyMeshRef = useRef<THREE.Points | null>(null);
  const ornamentManagersRef = useRef<MorphingInstances[]>([]);
  const ribbonManagerRef = useRef<MorphingInstances | null>(null);
  
  const photoGroupRef = useRef<THREE.Group | null>(null);
  const photoDataRef = useRef<PhotoData[]>([]);
  const billboardHelperRef = useRef<THREE.Object3D | null>(null);
  
  const focusStateRef = useRef({ active: false, id: null as number | null });
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const animationFrameIdRef = useRef<number>(0);
  
  const dragRef = useRef({
      isDragging: false,
      startX: 0,
      lastX: 0,
      lastY: 0,
      yawTarget: 0,
      pitchTarget: 0,
      rotationY: 0,
      rotationX: 0
  });

  const handTrackingRef = useRef<{x: number, y: number} | null>(null);
  const gestureState = useRef({
    lastRawGesture: 'UNKNOWN' as GestureType,
    debounceCount: 0,
    currentStableGesture: 'UNKNOWN' as GestureType,
    lastActionTime: 0
  });

  // Handle WISH mode looping logic
  useEffect(() => {
    let interval: any;
    if (debugGesture === 'V_SIGN') {
      setWishIndex(0);
      interval = setInterval(() => {
        setWishIndex(prev => (prev + 1) % WISH_LINES.length);
      }, 1500); // Transitions every 1.5 seconds
    } else {
      setWishIndex(0);
      if (interval) clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [debugGesture]);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020202);
    sceneRef.current = scene;

    const rootGroup = new THREE.Group();
    scene.add(rootGroup);
    sceneRootRef.current = rootGroup;
    
    const photoGroup = new THREE.Group();
    rootGroup.add(photoGroup);
    photoGroupRef.current = photoGroup;

    const billboardHelper = new THREE.Object3D();
    photoGroup.add(billboardHelper);
    billboardHelperRef.current = billboardHelper;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: "high-performance" });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    rendererRef.current = renderer;

    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;

    scene.add(new THREE.AmbientLight(0xffffff, 0.15));
    scene.add(new THREE.HemisphereLight(0xffffff, 0x080820, 0.3));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
    dirLight.position.set(5, 12, 8);
    scene.add(dirLight);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 1.5, 22);
    camera.lookAt(0, 1, 0);
    cameraRef.current = camera;

    const BLOOM_LAYER = 1;
    const darkMaterial = new THREE.MeshBasicMaterial({ color: 'black' });
    const materials: { [uuid: string]: THREE.Material | THREE.Material[] } = {};
    const renderPass = new RenderPass(scene, camera);
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.4, 0.2, 0.9);
    bloomPassRef.current = bloomPass;
    const bloomComposer = new EffectComposer(renderer);
    bloomComposer.renderToScreen = false;
    bloomComposer.addPass(renderPass);
    bloomComposer.addPass(bloomPass);
    const mixPass = new ShaderPass(
      new THREE.ShaderMaterial({
        uniforms: { baseTexture: { value: null }, bloomTexture: { value: (bloomComposer as any).renderTarget1.texture } },
        vertexShader: mixVertexShader, fragmentShader: mixFragmentShader
      }), 'baseTexture'
    );
    const finalComposer = new EffectComposer(renderer);
    finalComposer.addPass(renderPass);
    finalComposer.addPass(mixPass);
    finalComposer.addPass(new OutputPass());

    const sphereGeo = new THREE.SphereGeometry(0.12, 16, 16);
    const physicalParams = { metalness: 0.9, roughness: 0.15, clearcoat: 1.0, envMapIntensity: 1.2 };
    
    const createGroup = (color: number, count: number) => {
        const mat = new THREE.MeshPhysicalMaterial({ ...physicalParams, color, emissive: color, emissiveIntensity: 0.1 });
        const mesh = new THREE.InstancedMesh(sphereGeo, mat, count);
        const mgr = new MorphingInstances(mesh, count);
        const treeHeight = 12, treeBaseWidth = 4.2, treeBottomY = -5.0;
        for(let i=0; i<count; i++) {
            const hPercent = Math.random();
            const ty = treeBottomY + hPercent * treeHeight;
            const r = (1 - hPercent) * treeBaseWidth * (0.92 + 0.1 * Math.random());
            const theta = Math.random() * Math.PI * 2;
            mgr.treePositions.push(new THREE.Vector3(r * Math.cos(theta), ty, r * Math.sin(theta)));
            const isCore = Math.random() < 0.70;
            const { pos, isHalo } = getIsotropicScatterPos(isCore);
            mgr.scatterPositions.push(pos);
            mgr.haloFlags.push(isHalo);
            mgr.dummy.position.copy(mgr.treePositions[i]);
            mgr.dummy.updateMatrix();
            mgr.mesh.setMatrixAt(i, mgr.dummy.matrix);
        }
        rootGroup.add(mesh);
        return mgr;
    };
    ornamentManagersRef.current = [
        createGroup(0xC1121F, 80),
        createGroup(0xFDF0D5, 60),
        createGroup(0x669BBC, 40)
    ];

    const ribbonCount = 1000;
    const ribbonMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.02, 8, 8), new THREE.MeshBasicMaterial({ color: 0xFFD700 }), ribbonCount);
    ribbonMesh.layers.enable(BLOOM_LAYER);
    const ribbonManager = new MorphingInstances(ribbonMesh, ribbonCount);
    const treeHeight = 12, treeBaseWidth = 4.2, treeBottomY = -5.0;
    for(let i=0; i<ribbonCount; i++) {
       const t = i / ribbonCount; 
       const ty = treeBottomY + t * treeHeight;
       const r = (1 - t) * treeBaseWidth * 1.06; 
       const theta = t * Math.PI * 2 * 9.5;
       ribbonManager.treePositions.push(new THREE.Vector3(r * Math.cos(theta), ty, r * Math.sin(theta)));
       const isCore = Math.random() < 0.60;
       const { pos, isHalo } = getIsotropicScatterPos(isCore);
       ribbonManager.scatterPositions.push(pos);
       ribbonManager.haloFlags.push(isHalo);
       ribbonManager.dummy.position.copy(ribbonManager.treePositions[i]);
       ribbonManager.dummy.updateMatrix();
       ribbonMesh.setMatrixAt(i, ribbonManager.dummy.matrix);
    }
    rootGroup.add(ribbonMesh);
    ribbonManagerRef.current = ribbonManager;

    const clock = new THREE.Clock();
    let currentMorph = 0;
    (canvas as any).stateRef = { target: 0 };

    const bloomLayer = new THREE.Layers(); bloomLayer.set(BLOOM_LAYER);
    const darken = (o: THREE.Object3D) => { if ((o as any).isMesh && !o.layers.test(bloomLayer)) { materials[o.uuid] = (o as any).material; (o as any).material = darkMaterial; } };
    const restore = (o: THREE.Object3D) => { if (materials[o.uuid]) { (o as any).material = materials[o.uuid]; delete materials[o.uuid]; } };

    let requestID: number;
    const animate = () => {
      requestID = requestAnimationFrame(animate);
      const dt = Math.min(clock.getDelta(), 0.05);
      const time = clock.getElapsedTime();
      const target = (canvas as any).stateRef.target;
      
      if (Math.abs(target - currentMorph) > 0.001) currentMorph += (target - currentMorph) * 0.05;
      else currentMorph = target;

      const drag = dragRef.current;
      drag.rotationY += (drag.yawTarget - drag.rotationY) * 0.1;
      drag.rotationX += (drag.pitchTarget - drag.rotationX) * 0.1;
      rootGroup.rotation.y = drag.rotationY + time * AUTO_ROT_SPEED; 
      rootGroup.rotation.x = drag.rotationX;
      rootGroup.updateMatrixWorld();

      const photos = photoDataRef.current;
      const focusState = focusStateRef.current;

      if (focusState.active && focusState.id === null && photos.length > 0 && cameraRef.current) {
          let bestDist = Infinity;
          let bestId = -1;
          const camPos = cameraRef.current.position;
          const tempVec = new THREE.Vector3();
          photos.forEach(p => {
              p.mesh.getWorldPosition(tempVec);
              const d = tempVec.distanceTo(camPos);
              if(d < bestDist) { bestDist = d; bestId = p.idx; }
          });
          if (bestId !== -1) focusState.id = bestId;
      } else if (!focusState.active) {
          focusState.id = null;
      }

      if (photos.length > 0) {
          const easedMorph = currentMorph < 0.5 ? 4.0 * currentMorph * currentMorph * currentMorph : 1.0 - Math.pow(-2.0 * currentMorph + 2.0, 3.0) / 2.0;
          const bbHelper = billboardHelperRef.current;
          photos.forEach((p) => {
              let targetPos = new THREE.Vector3();
              let targetScale = p.baseScale.clone();
              if (p.idx === focusState.id && cameraRef.current && photoGroupRef.current) {
                  const cam = cameraRef.current;
                  const worldTarget = cam.position.clone().add(cam.getWorldDirection(new THREE.Vector3()).multiplyScalar(4.0));
                  photoGroupRef.current.worldToLocal(worldTarget);
                  targetPos.copy(worldTarget);
                  targetScale.multiplyScalar(2.5);
              } else {
                  targetPos.lerpVectors(p.treePos, p.scatterPos, easedMorph);
                  targetPos.y += Math.sin(time * 0.5 + p.idx) * 0.08;
              }
              p.mesh.position.lerp(targetPos, PHOTO_DAMP);
              p.mesh.scale.lerp(targetScale, PHOTO_DAMP);
              if (cameraRef.current && bbHelper) {
                  bbHelper.position.copy(p.mesh.position);
                  bbHelper.lookAt(cameraRef.current.position);
                  p.mesh.quaternion.slerp(bbHelper.quaternion, ROT_DAMP);
              }
          });
      }

      ornamentManagersRef.current.forEach(mgr => { mgr.update(currentMorph, time); });
      if (ribbonManagerRef.current) { ribbonManagerRef.current.update(currentMorph, time); }
      if (bodyMeshRef.current) {
        const m = bodyMeshRef.current;
        (m.material as THREE.ShaderMaterial).uniforms.uTime.value = time;
        (m.material as THREE.ShaderMaterial).uniforms.uMorph.value = currentMorph;
      }
      scene.traverse(darken); bloomComposer.render(); scene.traverse(restore);
      finalComposer.render();
    };
    animate();

    const handleResize = () => {
      const w = window.innerWidth, h = window.innerHeight;
      camera.aspect = w / h; camera.updateProjectionMatrix();
      renderer.setSize(w, h); bloomComposer.setSize(w, h); finalComposer.setSize(w, h);
      if (bodyMeshRef.current) (bodyMeshRef.current.material as THREE.ShaderMaterial).uniforms.uSizeScale.value = h / 45.0;
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(requestID);
      renderer.dispose();
      pmremGenerator.dispose();
    };
  }, []);

  const handlePhotoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (!e.target.files || !photoGroupRef.current) return;
      const files = Array.from(e.target.files).slice(0, 12) as File[];
      photoDataRef.current.forEach(p => {
          photoGroupRef.current?.remove(p.mesh);
          p.mesh.geometry.dispose();
          if (Array.isArray(p.mesh.material)) {
              p.mesh.material.forEach(m => (m as any).map?.dispose());
              p.mesh.material.forEach(m => m.dispose());
          } else {
              (p.mesh.material as any).map?.dispose();
              (p.mesh.material as THREE.Material).dispose();
          }
      });
      photoDataRef.current = [];
      const loader = new THREE.TextureLoader();
      const treeHeight = 12, treeBaseWidth = 4.2, treeBottomY = -5.0;
      files.forEach((file, idx) => {
          const url = URL.createObjectURL(file);
          loader.load(url, (tex) => {
             tex.colorSpace = THREE.SRGBColorSpace;
             const geometry = new THREE.PlaneGeometry(1, 1);
             const material = new THREE.MeshBasicMaterial({ 
                 map: tex, transparent: true, side: THREE.DoubleSide, depthWrite: false, toneMapped: false, opacity: 1.0 
             });
             const mesh = new THREE.Mesh(geometry, material);
             const aspect = tex.image.width / tex.image.height;
             const scaleVec = new THREE.Vector3(0.65 * aspect, 0.65, 1);
             mesh.scale.copy(scaleVec);
             const t = idx / files.length;
             const ty = treeBottomY + t * treeHeight;
             const tr = (1 - t) * treeBaseWidth * 1.2;
             const theta = t * Math.PI * 2 * 9.5;
             // Fixed error: replaced 'r' with 'tr' which is the correct radius variable defined above.
             const treePos = new THREE.Vector3(tr * Math.cos(theta), ty, tr * Math.sin(theta));
             const rMin = SCATTER_CONFIG.CORE_RADIUS * 1.25, rMax = SCATTER_CONFIG.OUTER_RADIUS;
             const sr = Math.cbrt(Math.random() * (Math.pow(rMax, 3) - Math.pow(rMin, 3)) + Math.pow(rMin, 3));
             const u = Math.random(), v = Math.random();
             const sTheta = 2 * Math.PI * u, sPhi = Math.acos(2 * v - 1);
             const scatterPos = new THREE.Vector3(sr * Math.sin(sPhi) * Math.cos(sTheta), sr * Math.sin(sPhi) * Math.sin(sTheta) + SCATTER_CENTER_Y, sr * Math.cos(sPhi));
             mesh.position.copy(treePos);
             photoGroupRef.current?.add(mesh);
             photoDataRef.current.push({ mesh, treePos, scatterPos, idx, baseScale: scaleVec });
          });
      });
  };

  useEffect(() => {
    if (!sceneRef.current || !sceneRootRef.current) return;
    let count = perfMode === 'High' ? 6000 : perfMode === 'Medium' ? 4000 : 2000;
    if (bloomPassRef.current) bloomPassRef.current.strength = perfMode === 'High' ? 0.45 : perfMode === 'Medium' ? 0.35 : 0.25;
    if (bodyMeshRef.current) {
      sceneRootRef.current.remove(bodyMeshRef.current);
      bodyMeshRef.current.geometry.dispose();
      (bodyMeshRef.current.material as THREE.Material).dispose();
      bodyMeshRef.current = null;
    }
    const bodyGeo = new THREE.BufferGeometry();
    const bodyPos = new Float32Array(count * 3), bodyScatter = new Float32Array(count * 3), bodyColors = new Float32Array(count * 3);
    const bodySpark = new Float32Array(count), bodySeed = new Float32Array(count), bodyHalo = new Float32Array(count);
    const treeHeight = 12, treeBaseWidth = 4.2, treeBottomY = -5.0;
    for (let i = 0; i < count; i++) {
      const hBias = Math.pow(Math.random(), 2.0); 
      const ty = treeBottomY + hBias * treeHeight;
      const r = (1 - hBias) * treeBaseWidth * (0.86 + 0.14 * Math.random()) * (0.90 + 0.10 * Math.sin(ty * 4.0 + Math.random()*6.0));
      const theta = Math.random() * Math.PI * 2;
      bodyPos[i*3] = r * Math.cos(theta); bodyPos[i*3+1] = ty; bodyPos[i*3+2] = r * Math.sin(theta);
      const isCore = Math.random() < SCATTER_CONFIG.CORE_RATIO;
      const { pos, isHalo } = getIsotropicScatterPos(isCore);
      bodyScatter[i*3] = pos.x; bodyScatter[i*3+1] = pos.y; bodyScatter[i*3+2] = pos.z;
      bodyHalo[i] = isHalo ? 1.0 : 0.0;
      bodyColors[i*3] = 0.1; bodyColors[i*3+1] = 0.6 + Math.random() * 0.3; bodyColors[i*3+2] = 0.2;
      bodySpark[i] = (Math.random() > 0.9) ? 1.0 : 0.0;
      bodySeed[i] = Math.random();
    }
    bodyGeo.setAttribute('position', new THREE.BufferAttribute(bodyPos, 3));
    bodyGeo.setAttribute('aScatterPosition', new THREE.BufferAttribute(bodyScatter, 3));
    bodyGeo.setAttribute('aColor', new THREE.BufferAttribute(bodyColors, 3));
    bodyGeo.setAttribute('aSpark', new THREE.BufferAttribute(bodySpark, 1));
    bodyGeo.setAttribute('aSeed', new THREE.BufferAttribute(bodySeed, 1));
    bodyGeo.setAttribute('aHalo', new THREE.BufferAttribute(bodyHalo, 1));
    const bodyMat = new THREE.ShaderMaterial({
      vertexShader: bodyVertexShader, fragmentShader: bodyFragmentShader,
      uniforms: { uTime: { value: 0 }, uMorph: { value: 0 }, uSizeScale: { value: window.innerHeight / 45.0 }, uOpacity: { value: 0.4 } },
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending
    });
    const bodyMesh = new THREE.Points(bodyGeo, bodyMat);
    sceneRootRef.current.add(bodyMesh);
    bodyMeshRef.current = bodyMesh;
  }, [perfMode]);

  const handlePointerDown = (e: React.PointerEvent) => {
      if ((e.target as HTMLElement).closest('button')) return;
      dragRef.current.isDragging = true;
      dragRef.current.lastX = e.clientX;
      dragRef.current.lastY = e.clientY;
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };
  const handlePointerMove = (e: React.PointerEvent) => {
      if (!dragRef.current.isDragging) return;
      dragRef.current.yawTarget += (e.clientX - dragRef.current.lastX) * 0.005;
      dragRef.current.pitchTarget += (e.clientY - dragRef.current.lastY) * 0.005;
      dragRef.current.pitchTarget = Math.max(-0.6, Math.min(0.6, dragRef.current.pitchTarget));
      dragRef.current.lastX = e.clientX;
      dragRef.current.lastY = e.clientY;
  };
  const handlePointerUp = (e: React.PointerEvent) => {
      dragRef.current.isDragging = false;
      (e.target as HTMLElement).releasePointerCapture(e.pointerId);
  };

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => { if (e.key.toLowerCase() === 'h') setUiVisible(prev => !prev); };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);

  useEffect(() => { if (canvasRef.current) (canvasRef.current as any).stateRef.target = targetMorph; }, [targetMorph]);

  const classifyGesture = (landmarks: NormalizedLandmark[]): GestureType => {
    const dist = (i: number, j: number) => Math.sqrt(Math.pow(landmarks[i].x - landmarks[j].x, 2) + Math.pow(landmarks[i].y - landmarks[j].y, 2));
    const isExtended = [dist(0, 4) > dist(0, 3) * 1.1, dist(0, 8) > dist(0, 6) * 1.1, dist(0, 12) > dist(0, 10) * 1.1, dist(0, 16) > dist(0, 14) * 1.1, dist(0, 20) > dist(0, 18) * 1.1];
    if (!isExtended[1] && !isExtended[2] && !isExtended[3] && !isExtended[4]) return 'FIST';
    if (isExtended[1] && isExtended[2] && isExtended[3] && isExtended[4]) return 'OPEN_PALM';
    if (isExtended[1] && isExtended[2] && !isExtended[3] && !isExtended[4]) return 'V_SIGN';
    if (isExtended[1] && !isExtended[2] && !isExtended[3] && !isExtended[4]) return 'INDEX_UP';
    return 'UNKNOWN';
  };

  useEffect(() => {
    setHandStatus('LOADING');
    const initHands = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm");
        handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`, delegate: "GPU" },
          runningMode: "VIDEO", numHands: 1
        });
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) { videoRef.current.srcObject = stream; videoRef.current.addEventListener('loadeddata', predictWebcam); setHandStatus('ON'); }
      } catch (err) { console.error(err); setHandStatus('ERROR'); }
    };
    const predictWebcam = () => {
        const video = videoRef.current; const canvas = handCanvasRef.current; const landmarker = handLandmarkerRef.current;
        if (!video || !canvas || !landmarker) return;
        if (video.videoWidth > 0) {
            canvas.width = video.videoWidth; canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d'); const now = performance.now();
            if (ctx) {
                const results = landmarker.detectForVideo(video, now);
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                if (results.landmarks && results.landmarks.length > 0) {
                    const landmarks = results.landmarks[0]; const rawGesture = classifyGesture(landmarks as NormalizedLandmark[]);
                    const wrist = landmarks[0];
                    if (handTrackingRef.current) {
                        const dx = Math.abs(wrist.x - handTrackingRef.current.x) < 0.003 ? 0 : wrist.x - handTrackingRef.current.x;
                        const dy = Math.abs(wrist.y - handTrackingRef.current.y) < 0.003 ? 0 : wrist.y - handTrackingRef.current.y;
                        if (dx !== 0 || dy !== 0) { setIsHandRotating(true); dragRef.current.yawTarget += dx * Math.PI * 1.25; dragRef.current.pitchTarget += dy * Math.PI * 0.6; dragRef.current.pitchTarget = Math.max(-0.6, Math.min(0.6, dragRef.current.pitchTarget)); }
                        else { setIsHandRotating(false); }
                    }
                    handTrackingRef.current = { x: wrist.x, y: wrist.y };
                    const st = gestureState.current;
                    if (rawGesture === st.lastRawGesture && rawGesture !== 'UNKNOWN') st.debounceCount++; else { st.debounceCount = 0; st.lastRawGesture = rawGesture; }
                    if (st.debounceCount >= 8) {
                        st.currentStableGesture = rawGesture; setDebugGesture(rawGesture);
                        focusStateRef.current.active = (st.currentStableGesture === 'INDEX_UP');
                        if (now - st.lastActionTime > 800) {
                             if (rawGesture === 'FIST') { setTargetMorph(0); st.lastActionTime = now; }
                             else if (rawGesture === 'OPEN_PALM') { setTargetMorph(1); st.lastActionTime = now; }
                             // WISH mode handled in separate useEffect
                        }
                    }
                    ctx.fillStyle = '#10b981'; ctx.shadowBlur = 10;
                    for (const point of landmarks) { ctx.beginPath(); ctx.arc(point.x * canvas.width, point.y * canvas.height, 3, 0, 2 * Math.PI); ctx.fill(); }
                } else { handTrackingRef.current = null; setIsHandRotating(false); focusStateRef.current.active = false; }
            }
        }
        animationFrameIdRef.current = requestAnimationFrame(predictWebcam);
    };
    initHands();
    return () => { cancelAnimationFrame(animationFrameIdRef.current); if (handLandmarkerRef.current) handLandmarkerRef.current.close(); };
  }, []);

  return (
    <div 
      className="relative w-screen h-screen bg-black overflow-hidden font-sans select-none text-white touch-none"
      onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={handlePointerUp} onPointerLeave={handlePointerUp}
    >
      <canvas ref={canvasRef} className="block w-full h-full outline-none" />
      <input type="file" multiple accept="image/*" className="hidden" ref={fileInputRef} onChange={handlePhotoUpload} />

      {/* Persistent UI Toggles - Top Right */}
      <div className="absolute top-8 right-8 z-50">
        <button
          onClick={() => setUiVisible(!uiVisible)}
          className="bg-white/5 hover:bg-white/10 border border-white/10 text-white/40 hover:text-white px-5 py-2 rounded-full text-[10px] tracking-[0.2em] uppercase font-bold backdrop-blur-md transition-all shadow-xl"
        >
          {uiVisible ? 'Hide UI' : 'Show UI'}
        </button>
      </div>

      {/* 1. Header Title - Fixed Center Top (Persistent Overlay - Always Visible) */}
      <div className="absolute top-[6%] left-1/2 -translate-x-1/2 z-30 pointer-events-none flex flex-col items-center">
        <h1 className="flex flex-col items-center text-xl md:text-2xl font-bold tracking-tighter text-emerald-200/90 uppercase text-center leading-[0.85]">
          <span>HOENERGY</span>
          <span className="mt-0.5">CHRISTMAS TREE</span>
        </h1>
      </div>

      {/* Hideable UI Container Start */}
      <div className={`transition-all duration-700 ${uiVisible ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
        
        {/* 2. Slim Gesture Hint Bar - Bottom Centered */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-40">
          <div className="flex items-center gap-6 bg-white/5 backdrop-blur-xl px-8 py-2.5 rounded-full border border-white/10 shadow-[0_4px_30px_rgba(0,0,0,0.5)]">
              {[
                { icon: '✊', label: 'Tree' },
                { icon: '✋', label: 'Scatter' },
                { icon: '☝️', label: 'Focus' },
                { icon: '✌️', label: 'Wish' }
              ].map(item => (
                <div key={item.label} className="flex items-center gap-2">
                  <span className="text-lg drop-shadow-sm">{item.icon}</span>
                  <span className="text-[10px] font-bold tracking-[0.2em] text-emerald-100/60 uppercase">
                    {item.label}
                  </span>
                </div>
              ))}
          </div>
        </div>

        {/* 3. Action Buttons - Bottom Left */}
        <div className="absolute bottom-8 left-8 z-30">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="bg-white/5 hover:bg-white/10 border border-white/10 text-white/80 hover:text-white px-8 py-3 rounded-full text-[10px] tracking-[0.2em] uppercase font-bold backdrop-blur-md transition-all shadow-xl"
          >
            Upload 12 Photos
          </button>
        </div>

        {/* 4. Minimal Camera Preview - Bottom Right */}
        <div className="absolute bottom-8 right-8 z-30 flex flex-col items-end gap-3">
           {isHandRotating && (
               <div className="text-[9px] text-emerald-400/80 font-bold tracking-[0.3em] uppercase animate-pulse">
                   Active
               </div>
           )}
           <div className="relative w-36 h-28 md:w-44 md:h-34 bg-black/40 rounded-[1.5rem] overflow-hidden border border-white/5 shadow-2xl transform scale-x-[-1]">
               <video ref={videoRef} autoPlay playsInline className="absolute w-full h-full object-cover opacity-20" />
               <canvas ref={handCanvasRef} className="absolute w-full h-full object-cover" />
           </div>
           <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-black/20 border border-white/5 backdrop-blur-md">
              <span className={`w-1 h-1 rounded-full ${handStatus === 'ON' ? 'bg-emerald-400/60' : 'bg-amber-400/60 animate-pulse'}`} />
              <span className="text-[8px] text-white/40 font-bold tracking-widest uppercase">
                 {handStatus === 'ON' ? debugGesture : 'Syncing'}
              </span>
           </div>
        </div>
      </div>
      {/* Hideable UI Container End */}

      {/* Magical WISH Text Loop - Centered */}
      {debugGesture === 'V_SIGN' && (
        <div key={wishIndex} className="absolute top-[45%] left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 animate-[fadeIn_0.5s_ease-out] pointer-events-none text-center">
          <h2 className="text-4xl md:text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-b from-yellow-100 to-white drop-shadow-[0_0_40px_rgba(253,224,71,0.5)] mb-3 italic">
             {WISH_LINES[wishIndex]}
          </h2>
        </div>
      )}

      {/* Subtle Bottom Toggle (Additional control) */}
      <button 
        onClick={() => setUiVisible(!uiVisible)} 
        className={`absolute bottom-2 left-1/2 -translate-x-1/2 z-50 opacity-10 hover:opacity-50 transition-opacity text-[8px] tracking-[0.4em] uppercase font-bold text-white/30 ${uiVisible ? 'hidden' : 'block'}`}
      >
        Show UI
      </button>

      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes fadeIn { from { opacity: 0; transform: translate(-50%, -40%); } to { opacity: 1; transform: translate(-50%, -50%); } }
      `}} />
    </div>
  );
};

export default App;
