"""SERUM diffusion teacher entrypoint sharing the audited LoRA training harness."""
import signal
from run import main

if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    main(method='serum')
