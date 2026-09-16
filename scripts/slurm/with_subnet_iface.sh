#!/bin/bash
# Run a command with NCCL and Gloo bound to the network interface that
# carries this node's address on MASTER_SUBNET (default 10.20.).
#
# Left to themselves NCCL and Gloo may pick an interface the other nodes
# cannot reach (e.g. a public address that is only routable from outside),
# and the process group then hangs while initializing. Interface names
# differ between nodes, so they cannot be set once for the whole job.
subnet=${MASTER_SUBNET:-10.20.}
ifc=$(ip -o -4 addr show | awk -v s="$subnet" 'index($4, s) == 1 {print $2; exit}')

if [ -n "$ifc" ]; then
    export NCCL_SOCKET_IFNAME=$ifc GLOO_SOCKET_IFNAME=$ifc
else
    echo "with_subnet_iface: no interface on ${subnet}* on $(hostname)" >&2
fi

exec "$@"
