document.addEventListener("alpine:init", () => {
  Alpine.data("dashboard", () => ({
    status: {
      mode: "idle", timesteps: 0, episodes: 0, current_x: 0,
      max_x_this_ep: 0, max_x_historical: 0, reward_avg_100: 0,
      distance_avg_100: 0, clear_rate_100: 0, coins_avg_100: 0,
      behavior_score: 0, last_episode_reward: 0, last_clear: false,
      paused: false, stopping: false
    },
    xMax: 3360,
    startReq: { config_path: "configs/default.yaml", resume_path: "" },
    modelPath: "models_saved/mario_ppo",
    log: "",
    rewardEdit: {},
    rewardFields: [
      { key: "forward_reward_coef", type: "number" },
      { key: "backward_reward_coef", type: "number" },
      { key: "coin_reward", type: "number" },
      { key: "score_reward_coef", type: "number" },
      { key: "vertical_explore_reward", type: "number" },
      { key: "time_penalty_coef", type: "number" },
      { key: "time_remaining_bonus_coef", type: "number" },
      { key: "flag_bonus", type: "number" },
      { key: "death_by_enemy_penalty", type: "number" },
      { key: "death_by_time_penalty", type: "number" },
      { key: "stuck_penalty_base", type: "number" },
      { key: "enable_cyclic_detection", type: "bool" },
      { key: "cyclic_action_penalty", type: "number" },
      { key: "enable_death_map", type: "bool" },
      { key: "death_bucket_coef", type: "number" },
      { key: "death_bucket_cap", type: "number" },
      { key: "enable_excessive_left", type: "bool" },
      { key: "excessive_left_penalty", type: "number" },
      { key: "enable_micro_movement", type: "bool" },
      { key: "micro_movement_penalty", type: "number" },
      { key: "enable_wall_stuck", type: "bool" },
      { key: "wall_stuck_penalty", type: "number" },
      { key: "enable_records", type: "bool" },
      { key: "record_distance_bonus", type: "number" },
      { key: "record_time_bonus", type: "number" }
    ],
    trajectories: [],
    charts: {},
    rewardSeries: [],
    clearSeries: [],

    async init() {
      await this.refreshConfig();
      this.connectMetrics();
      this.connectFrame();
      this.connectTrajectories();
      this.setupCharts();
    },

    async refreshConfig() {
      const r = await fetch("/config");
      const data = await r.json();
      this.xMax = data.x_max_level;
      const reward = data.config.reward;
      for (const field of this.rewardFields) {
        this.rewardEdit[field.key] = reward[field.key];
      }
    },

    connectMetrics() {
      const ws = new WebSocket(`ws://${location.host}/stream/metrics`);
      ws.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        Object.assign(this.status, data);
        this.pushChart(data);
      };
      ws.onclose = () => setTimeout(() => this.connectMetrics(), 2000);
    },

    connectFrame() {
      const img = document.getElementById("live-frame");
      const ws = new WebSocket(`ws://${location.host}/stream/frame`);
      ws.onmessage = (ev) => {
        img.src = "data:image/jpeg;base64," + ev.data;
      };
      ws.onclose = () => setTimeout(() => this.connectFrame(), 2000);
    },

    connectTrajectories() {
      const ws = new WebSocket(`ws://${location.host}/stream/trajectories`);
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.replace) {
          this.trajectories = msg.trajectories;
        } else if (msg.append) {
          this.trajectories.push(...msg.trajectories);
          if (this.trajectories.length > 50) {
            this.trajectories = this.trajectories.slice(-50);
          }
        }
        this.drawTrajectories();
      };
      ws.onclose = () => setTimeout(() => this.connectTrajectories(), 2000);
    },

    drawTrajectories() {
      const canvas = document.getElementById("level-canvas");
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;
      ctx.fillStyle = "#0c0f17";
      ctx.fillRect(0, 0, w, h);
      // grid
      ctx.strokeStyle = "#1a1f2c";
      for (let i = 0; i <= 10; i++) {
        const x = (i / 10) * w;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      const yMin = 50, yMax = 250;
      this.trajectories.forEach((traj, idx) => {
        const age = (this.trajectories.length - idx) / this.trajectories.length;
        const alpha = Math.max(0.15, 1.0 - age * 0.8);
        const hue = (idx * 47) % 360;
        ctx.strokeStyle = `hsla(${hue}, 80%, 60%, ${alpha})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        traj.forEach((pt, i) => {
          const x = (pt[0] / this.xMax) * w;
          const y = h - ((pt[1] - yMin) / (yMax - yMin)) * h;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });
    },

    setupCharts() {
      const mk = (id, label, color) => new Chart(document.getElementById(id), {
        type: "line",
        data: { labels: [], datasets: [{ label, data: [], borderColor: color, tension: 0.2, pointRadius: 0 }] },
        options: { animation: false, responsive: true, scales: {
          x: { ticks: { color: "#8a92a8" }, grid: { color: "#1a1f2c" } },
          y: { ticks: { color: "#8a92a8" }, grid: { color: "#1a1f2c" } }
        }, plugins: { legend: { labels: { color: "#aab0c2" } } } }
      });
      this.charts.reward = mk("chart-reward", "reward_avg_100", "#22c55e");
      this.charts.clear = mk("chart-clear", "clear_rate_100 (%)", "#3b82f6");
    },

    pushChart(data) {
      if (!this.charts.reward) return;
      const ts = data.timesteps;
      const cr = this.charts.reward;
      const cc = this.charts.clear;
      cr.data.labels.push(ts); cc.data.labels.push(ts);
      cr.data.datasets[0].data.push(data.reward_avg_100);
      cc.data.datasets[0].data.push(data.clear_rate_100 * 100);
      const limit = 200;
      if (cr.data.labels.length > limit) {
        cr.data.labels.splice(0, cr.data.labels.length - limit);
        cc.data.labels.splice(0, cc.data.labels.length - limit);
        cr.data.datasets[0].data.splice(0, cr.data.datasets[0].data.length - limit);
        cc.data.datasets[0].data.splice(0, cc.data.datasets[0].data.length - limit);
      }
      cr.update("none"); cc.update("none");
    },

    async trainStart() {
      const body = {
        config_path: this.startReq.config_path,
        resume_path: this.startReq.resume_path || null
      };
      await this.post("/train/start", body);
    },
    async trainPause() { await this.post("/train/pause", {}); },
    async trainResume() { await this.post("/train/resume", {}); },
    async trainStop() { await this.post("/train/stop", {}); },
    async modelSave() { await this.post("/model/save", { path: this.modelPath }); },
    async modelLoad() { await this.post("/model/load", { path: this.modelPath }); },

    async applyReward() {
      const body = { reward: {} };
      for (const field of this.rewardFields) {
        body.reward[field.key] = this.rewardEdit[field.key];
      }
      const r = await fetch("/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const data = await r.json();
      this.log = "aplicados: " + data.applied.join(", ");
    },

    async post(url, body) {
      try {
        const r = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        const data = await r.json().catch(() => ({}));
        this.log = (data && data.message) ? data.message : (r.ok ? "ok" : "error");
      } catch (e) {
        this.log = "error: " + e;
      }
    }
  }));
});
