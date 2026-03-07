from __future__ import annotations
import hashlib
from pathlib import Path

def load_fmu(fmu_path: str, start_time: float = 0.0, **kwargs):
    """
    Minimal FMI 2.0 Co-Simulation loader using fmpy, with a pyfmi-like API:
      - set(names, values)
      - get(names)
      - do_step(current_t, step_size)
      - setup_experiment(start_time)
      - initialize()
      - reset()

    Your FMU is Co-Simulation only (md.coSimulation True, md.modelExchange False).
    """
    return _FMPYCoSim(fmu_path, start_time=start_time)

def _cache_extract_dir(fmu_path: str) -> str:
    p = Path(fmu_path).resolve()
    key = hashlib.sha1((str(p) + str(p.stat().st_mtime_ns)).encode()).hexdigest()[:16]
    d = Path.home() / ".cache" / "sustainlc_fmu" / key
    d.mkdir(parents=True, exist_ok=True)
    return str(d)

class _FMPYCoSim:
    def __init__(self, fmu_path: str, start_time: float = 0.0):
        from fmpy import read_model_description, extract
        from fmpy.fmi2 import FMU2Slave

        self.fmu_path = fmu_path
        self.md = read_model_description(fmu_path)
        if self.md.coSimulation is None:
            raise RuntimeError("FMU has no Co-Simulation interface.")

        self.unzipdir = _cache_extract_dir(fmu_path)
        extract(fmu_path, self.unzipdir)

        self._slave = FMU2Slave(
            guid=self.md.guid,
            unzipDirectory=self.unzipdir,
            modelIdentifier=self.md.coSimulation.modelIdentifier,
            instanceName="sustainlc"
        )

        self._vr = {v.name: v.valueReference for v in self.md.modelVariables}
        self._typ = {}
        for v in self.md.modelVariables:
            t = v.type
            if getattr(t, "real", None) is not None:
                self._typ[v.name] = "Real"
            elif getattr(t, "integer", None) is not None:
                self._typ[v.name] = "Integer"
            elif getattr(t, "boolean", None) is not None:
                self._typ[v.name] = "Boolean"
            elif getattr(t, "string", None) is not None:
                self._typ[v.name] = "String"
            else:
                self._typ[v.name] = "Real"

        self._instantiated = False
        self._initialized = False
        self._t0 = float(start_time)

    def setup_experiment(self, start_time: float | None = None, stop_time: float | None = None):
        if not self._instantiated:
            self._slave.instantiate()
            self._instantiated = True
        t0 = self._t0 if start_time is None else float(start_time)
        if stop_time is None:
            self._slave.setupExperiment(startTime=t0)
        else:
            self._slave.setupExperiment(startTime=t0, stopTime=float(stop_time))

    def initialize(self):
        if not self._instantiated:
            self.setup_experiment()
        if not self._initialized:
            self._slave.enterInitializationMode()
            self._slave.exitInitializationMode()
            self._initialized = True

    def reset(self):
        self._slave.reset()
        self._initialized = False

    def do_step(self, current_t: float, step_size: float):
        self._slave.doStep(float(current_t), float(step_size))

    def set(self, names, values=None):
        if isinstance(names, dict):
            items = list(names.items())
            names = [k for k, _ in items]
            values = [v for _, v in items]
        elif isinstance(names, (str,)):
            names = [names]
            values = [values]

        groups = {"Real": [], "Integer": [], "Boolean": [], "String": []}
        for n, v in zip(list(names), list(values)):
            groups[self._typ[n]].append((self._vr[n], v))

        if groups["Real"]:
            vr, vv = zip(*groups["Real"])
            self._slave.setReal(list(vr), [float(x) for x in vv])
        if groups["Integer"]:
            vr, vv = zip(*groups["Integer"])
            self._slave.setInteger(list(vr), [int(x) for x in vv])
        if groups["Boolean"]:
            vr, vv = zip(*groups["Boolean"])
            self._slave.setBoolean(list(vr), [bool(x) for x in vv])
        if groups["String"]:
            vr, vv = zip(*groups["String"])
            self._slave.setString(list(vr), [str(x) for x in vv])

    def get(self, names):
        # returns a list for scalar name (pyfmi style): fmu.get("x")[0]
        if isinstance(names, (str,)):
            return [self._get_one(names)]
        out = []
        for n in list(names):
            out.append([self._get_one(n)])
        return out

    def _get_one(self, name: str):
        vr = [self._vr[name]]
        typ = self._typ[name]
        if typ == "Real":
            return self._slave.getReal(vr)[0]
        if typ == "Integer":
            return self._slave.getInteger(vr)[0]
        if typ == "Boolean":
            return self._slave.getBoolean(vr)[0]
        return self._slave.getString(vr)[0]
