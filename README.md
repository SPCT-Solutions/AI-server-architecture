# AI-server-architecture
# Bib Detection + OCR Pipeline: Architecture

Scope: Server 1 (CPU: orchestrator + YOLO), Server 2 (GPU: Gemma via llama.cpp), the Socket.IO broker between them, MySQL, and the PHP photo server.

Legend used in all diagrams:

| Colour | Meaning |
|---|---|
| Blue | CPU work |
| Green | GPU work |
| Amber | Disk / DB / network I/O |


---

## 1. Big picture: where CPU and GPU are used

```mermaid
flowchart LR
  CAM["photos uploads<br/>/BackupPhotos/YYYY/EVENT/*.JPG"]:::io

  subgraph S1["SERVER 1 - CPU ONLY (CUDA_VISIBLE_DEVICES empty, DEVICE=cpu)"]
    WATCH["Folder watcher<br/>watchdog thread + 1s loop"]:::cpu
    ORCH["AI_File.py<br/>orchestrator, spawns workers"]:::cpu
    VL["AI_vL.yolo<br/>(local yolo model file)"]:::ext
    VO["AI_vO.process_api<br/>(open ai file)"]:::ext
    VG["AI_vG.process_api<br/>YOLO11x person + bib model<br/>on CPU, TTA on"]:::cpu
    DB[("MySQL spctAI<br/>images table")]:::io
  end

  HUB["Node Socket.IO broker<br/>127.0.0.1:1947 / public /socket.io/"]:::ext

  subgraph S2["SERVER 2 - GPU"]
    GEM["AI_Gemma.py<br/>1 process, 4 worker threads<br/>HTTP client + watchdog"]:::cpu
    LLAMA["llama-server (PM2)<br/>Gemma-4 e4b Q4_K_M + mmproj<br/>--parallel 4, -ngl 999, -c 4096"]:::gpu
  end

  PHP["photo2.spct.kr<br/>photoprocessorX.php"]:::ext

  CAM --> WATCH
  WATCH -->|"TCP localhost:65432<br/>JSON list"| ORCH
  DB -->|"SELECT Pending LIMIT 2000"| ORCH
  ORCH -->|"Process per batch of 50<br/>aiModel 0 / 1 / 2"| VL
  ORCH --> VO
  ORCH --> VG
  VG -->|"crop packets<br/>event processBibGemma4"| HUB
  HUB --> GEM
  GEM -->|"POST /v1/chat/completions<br/>base64 crop"| LLAMA
  LLAMA --> GEM
  GEM -->|"event processBibGemma4Done"| HUB
  HUB -->|"broadcast, PID filter"| VG
  VG -->|"HTTPS POST full image + bib"| PHP
  VG -->|"update_bulk"| DB

  classDef cpu fill:#dbeafe,stroke:#1d4ed8,color:#111;
  classDef gpu fill:#dcfce7,stroke:#15803d,color:#111;
  classDef io fill:#fef3c7,stroke:#b45309,color:#111;
  classDef ext fill:#f3f4f6,stroke:#6b7280,color:#111;
```

**Key fact:** the only GPU work in the whole system is `llama-server` on Server 2. All YOLO inference (person + bib, up to ~17 forward-pass groups per image) runs on Server 1 CPU.

---

## 2. Server 1: processes and threads

```mermaid
flowchart TB
  subgraph P0["PROCESS A: AI_File.py __main__ (spawn start method)"]
    A1["MAIN THREAD<br/>start_server(): blocking TCP accept loop :65432<br/>recv(204800) then json.loads then G_list.extend"]:::cpu
    A2["socketio client threads<br/>(emit 'reload' at startup)"]:::io
  end

  subgraph PM["PROCESS B: multiprocessing.Manager server"]
    B1[("G_list<br/>shared proxy list")]:::cpu
  end

  subgraph PC["PROCESS C: check_G_List loop"]
    C1["MAIN THREAD loop<br/>1. release finished workers<br/>2. if list empty and no workers: DB fetch Pending (2000)<br/>3. psutil.cpu_percent(interval=1) BLOCKS 1s<br/>4. if cpu under 40 and workers under 4: dispatch"]:::cpu
    C2["socketio client threads<br/>(system_info emit every loop)"]:::io
  end

  subgraph PW["PROCESSES D..: workers (new Process per batch, up to 3 per cycle)"]
    D1["AI_vL.yolo(batch of 50) - aiModel 0"]:::ext
    D2["AI_vO.process_api(batch of 50) - aiModel 1"]:::ext
    D3["AI_vG.process_api(batch of 50) - aiModel 2"]:::cpu
  end

  A1 <--> B1
  C1 <--> B1
  C1 -->|"multiprocessing.Process().start()"| D1
  C1 --> D2
  C1 --> D3

  classDef cpu fill:#dbeafe,stroke:#1d4ed8,color:#111;
  classDef io fill:#fef3c7,stroke:#b45309,color:#111;
  classDef ext fill:#f3f4f6,stroke:#6b7280,color:#111;
```

### Inside one `AI_vG` worker process

```mermaid
flowchart TB
  subgraph W["AI_vG worker process (one per batch of up to 50 images)"]
    M["MAIN THREAD<br/>loads yolo11x.pt + bibModelBest.pt (every batch)<br/>sequential for-loop over images<br/>ALL YOLO inference here"]:::cpu
    OMP["torch / OpenCV intra-op threads<br/>default = all cores (per process!)"]:::cpu
    S["socketio client threads<br/>engineio read + write loops"]:::io
    H["on_done handler thread(s)<br/>@processBibGemma4Done<br/>PID filter, cv2.imread, POST to PHP, DB bulk update"]:::cpu
    L["locks: done_lock, db_lock"]:::cpu
  end
  M --> OMP
  M -->|"emit crops"| S
  S -->|"Done event"| H
  H --> L

  classDef cpu fill:#dbeafe,stroke:#1d4ed8,color:#111;
  classDef io fill:#fef3c7,stroke:#b45309,color:#111;
```

---

## 3. Per-image pipeline in `AI_vG` (all CPU)

```mermaid
flowchart TB
  S0["cv2.imread full-res JPEG<br/>DISK I/O"]:::io
  S0b["image_to_base64(full image)<br/>JPEG encode, result UNUSED on main path"]:::cpu
  S1["STAGE 1: person detection<br/>yolo11x, conf 0.30, iou 0.60, max_det 15<br/>1 forward pass"]:::cpu
  D1{"any person?"}
  DEAD["dead_image(): emit status,<br/>re-read image, encode, POST to PHP"]:::io
  NMS["apply_person_nms (IoU 0.65), sort left to right"]:::cpu
  P1["STAGE 2 / PASS 1: bib model on full image<br/>augment=True (TTA ~3 passes)"]:::cpu
  P2["PASS 2: bib model on torso crop of EACH person<br/>K = up to 15 crops, augment=True each<br/>~3 x K forward passes"]:::cpu
  MERGE["merge raw bibs, IoU 0.30 dedup"]:::cpu
  D2{"any bib?"}
  HUN["Hungarian assignment (scipy)<br/>7-signal score, min 0.25"]:::cpu
  CROP["crop each assigned bib<br/>JPEG encode to base64"]:::cpu
  EMIT["emit processBibGemma4 per bib<br/>packet c of t, isDone flag<br/>sleep(2) after FINAL packet"]:::io

  S0 --> S0b --> S1 --> D1
  D1 -->|no| DEAD
  D1 -->|yes| NMS --> P1 --> P2 --> MERGE --> D2
  D2 -->|no| DEAD
  D2 -->|yes| HUN --> CROP --> EMIT

  classDef cpu fill:#dbeafe,stroke:#1d4ed8,color:#111;
  classDef io fill:#fef3c7,stroke:#b45309,color:#111;
```

**Forward-pass budget per image (CPU):** 1 person pass (yolo11x, largest YOLO11 variant) + 3 (full-image bib TTA) + 3 x K (crop bib TTA). With K = 5 runners that is about 19 passes per image, all serial in one thread.

---

## 4. Server 2: `AI_Gemma.py` and llama-server

```mermaid
flowchart TB
  subgraph G["AI_Gemma.py (single process)"]
    SIO["socketio client threads<br/>(engineio loops)"]:::io
    OND["on_data handler<br/>queue if restarting, else register task<br/>MAX_QUEUE 50 tasks"]:::cpu
    EX["ThreadPoolExecutor: 4 threads 'llamacpp'<br/>1 thread = 1 image (row_id)<br/>crops of one image processed SEQUENTIALLY, in packet order"]:::cpu
    RL["rate limiter<br/>0.2 s spacing, sleeps INSIDE _rate_lock<br/>+ request semaphore (4)"]:::cpu
    TEMP["nvidia-smi subprocess<br/>spawned PER REQUEST"]:::cpu
    HTTP["requests.post, timeout 90s<br/>retry x6, backoff 2..30 s"]:::io
    WD["gpu-watchdog thread<br/>every 15 s: nvidia-smi"]:::cpu
    CL["cleanup thread<br/>stale tasks over 300 s"]:::cpu
  end

  subgraph LS["llama-server (PM2, separate process)"]
    SL["4 slots (--parallel 4)<br/>context 4096 total = ~1024 per slot"]:::gpu
    VIS["mmproj vision encoder"]:::gpu
    LLM["Gemma-4 e4b Q4_K_M<br/>all layers on GPU, max_tokens 48"]:::gpu
  end

  SIO --> OND --> EX --> RL --> TEMP --> HTTP --> SL
  SL --> VIS --> LLM
  LLM --> HTTP
  EX -->|"all packets done"| SIO
  WD -.->|"pkill -9 + pm2 restart"| LS
  WD -.->|"sets error flag / queue"| OND

  classDef cpu fill:#dbeafe,stroke:#1d4ed8,color:#111;
  classDef gpu fill:#dcfce7,stroke:#15803d,color:#111;
  classDef io fill:#fef3c7,stroke:#b45309,color:#111;
```

### GPU watchdog rules

```mermaid
flowchart TB
  T["every 15 s: read temp + VRAM"]:::cpu
  R1{"VRAM >= 9000 MiB<br/>AND active (idle under 30 s)<br/>AND cooldown 300 s passed"}
  R2{"temp >= 82 C"}
  R3{"VRAM >= 11000 MiB"}
  R4{"idle >= 600 s AND VRAM >= 6000<br/>AND cooldown passed"}
  RS["_restart_llama_server<br/>pkill -9, pm2 restart,<br/>wait /health up to 60 s"]:::gpu
  T --> R1 -->|yes| RS
  T --> R2 -->|yes| RS
  T --> R3 -->|yes| RS
  T --> R4 -->|yes| RS

  classDef cpu fill:#dbeafe,stroke:#1d4ed8,color:#111;
  classDef gpu fill:#dcfce7,stroke:#15803d,color:#111;
```

Only the 9000 MiB and idle rules honour the 300 s cooldown. The temp and critical-VRAM rules can restart repeatedly.

---

## 5. End-to-end sequence for one image

```mermaid
sequenceDiagram
  autonumber
  participant DB as MySQL
  participant OR as AI_File (check_G_List)
  participant VG as AI_vG worker (CPU)
  participant HUB as Node broker
  participant GM as AI_Gemma (threads)
  participant LL as llama-server (GPU)
  participant PHP as photoprocessorX.php

  OR->>DB: SELECT Pending (aiModel 0,1,2) LIMIT 2000
  OR->>VG: spawn Process(batch of 50, aiModel 2)
  VG->>VG: imread, person YOLO, bib YOLO full + K crops, Hungarian
  loop for each assigned bib c = 1..t
    VG->>HUB: processBibGemma4 (crop b64, row, pid, c, t, isDone)
    HUB->>GM: broadcast
    GM->>GM: on_data, queue per row_id, in-order packets
    GM->>LL: POST chat/completions (one crop at a time per image)
    LL-->>GM: Bib: N / Name: X
  end
  GM->>HUB: processBibGemma4Done (bib list, name list, pid, row)
  HUB->>VG: broadcast to every vG worker
  VG->>VG: on_done: skip if pid is not mine
  VG->>VG: imread full image again, encode
  VG->>PHP: POST full image + bibno + name
  PHP-->>VG: status success
  VG->>HUB: updateById (UI log)
  VG->>DB: update_bulk (new connection)
```

---

## 6. Resource usage matrix

| Stage | Where | Resource | Threads / procs | Notes |
|---|---|---|---|---|
| Folder watch | Server 1 | CPU (light), inotify | 1 watchdog thread + main loop (1 s) | |
| DB poll | Server 1 `check_G_List` proc | MySQL | main thread | New connection each call, never closed |
| CPU gate | Server 1 `check_G_List` proc | CPU | main thread | `cpu_percent(interval=1)` blocks 1 s each loop |
| Image read | vG worker | Disk I/O | main thread | Full image read in worker AND again in `on_done` |
| Person YOLO11x | vG worker | **CPU** | main + OpenMP threads | Model loaded |
| Bib YOLO, full image, TTA | vG worker | **CPU** | main + OpenMP threads | ~3 passes |
| Bib YOLO, per-person crops, TTA | vG worker | **CPU** | main + OpenMP threads | ~3 x K passes, dominant cost |
| Hungarian assign | vG worker | CPU (tiny) | main | |
| Crop encode + emit | vG worker | CPU + network | main + sio threads | base64 JPEG through broker |
| Broker fan-out | Node | CPU + network | n/a | Broadcast, every client gets every packet |
| Packet routing | Gemma proc | CPU | sio thread + 4 pool threads | |
| nvidia-smi temp check | Gemma proc | CPU (process spawn) | per request | Once per crop |
| Vision encode + LLM decode | llama-server | **GPU** | 4 slots | Only GPU use in system |
| OCR result post | vG `on_done` thread | CPU + network | handler thread | Re-reads and encodes full image |
| DB bulk update | vG `on_done` thread | MySQL | handler thread | New connection per call |

---

## 7. Concurrency limits (the numbers that cap throughput)

| Knob | Value | Effect |
|---|---|---|
| `batchSize` | 50 per aiModel per cycle | Rows per worker process |
| Dispatch gate | `cpu < 40%` and `len(process_list) < 4` | Up to 3 new procs per cycle, so list can exceed 4. `MAX_PROCS`  |
| vG workers in parallel | up to ~3-6 total across vL/vO/vG | Each uses all cores for torch |
| `PERSON_MAX_DET` | 15 | Max crops per image |
| `WORKER_THREADS` (Gemma) | 4 | 4 images in parallel, 1 crop in flight each |
| `--parallel` (llama) | 4 | Matches the 4 worker threads |
| `MIN_REQUEST_INTERVAL` | 0.2 s | Caps request starts at 5/s |
| `MAX_QUEUE` | 50 tasks | Excess images are dropped with no Done event |
| `LLAMA_TIMEOUT` / retries | 90 s / 6 | Worst case per crop is minutes |
| `-c 4096`, parallel 4 | ~1024 ctx per slot | Prompt + image tokens + 48 out must fit |

---

## 8. Optimization candidates

| # | Lever | Where | Expected impact | Risk |
|---|---|---|---|---|
| 1 | YOLO inference to GPU (TensorRT/ONNX-GPU, ideally on  a GPU on Server 1) | vG |  CPU YOLO is the bottleneck | 
| 2 | Turn off `augment=True` on pass 2 (or both) | vG | ~3x fewer bib passes | Small recall drop, test on hard images |
| 3 | Smaller person model (yolo11m/l)  | vG | Large CPU saving | 
| 4 | Persistent worker pool, load models once | AI_File / vG | 
| 5 | Targeted emits / rooms instead of broadcast; avoid base64 over broker | Broker | Less network + parse load | Broker change |
| 6 | One shared MySQL connection (or pool), close after use | DB | Fewer leaked connections | None |
| 7 | Revisit `-c 4096` with 4 slots | llama | Avoid silent truncation | VRAM use |

---

