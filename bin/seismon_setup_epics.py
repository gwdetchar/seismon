import cas
CHANNELS = {
    'H1:AMP': {
        'type': 'float',
    },
    'L1:AMP': {
        'type': 'float',
    },
    'V1:AMP': {
        'type': 'float',
    },
    'G1:AMP': {
        'type': 'float',
    },

    'MIT:AMP': {
        'type': 'float',
    },
}
prefix = 'SEISMON:'
server = cas.CaServer(prefix, CHANNELS)
server.run()
