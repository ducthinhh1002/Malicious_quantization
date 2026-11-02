try:
 import setproctitle
 setproctitle.setproctitle('VLLM::EngineCore')
 setproctitle.setthreadtitle('VLLM::EngineCore')
except ImportError:
 pass
