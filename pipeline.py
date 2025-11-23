import os
import sys

NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 0))

print(f"[pipeline.py] Starting, NODE_NUMBER={NODE_NUMBER}")

if NODE_NUMBER == 0:
    print("[pipeline.py] Launching Node0 frontend...")
    import endpoint_node

    endpoint_node.main()

elif NODE_NUMBER == 1:
    print("[pipeline.py] Launching Node1 retrieval...")
    import rag_node

    rag_node.main()

elif NODE_NUMBER == 2:
    print("[pipeline.py] Launching Node2 generator...")
    import rag_node

    rag_node.main()

else:
    print(f"[pipeline.py] ERROR: Unknown NODE_NUMBER={NODE_NUMBER}")
    sys.exit(1)
