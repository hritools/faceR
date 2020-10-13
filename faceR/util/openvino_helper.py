def reshape(net, new_shape):
    net.reshape({new_shape['input name']: new_shape['input shape']})


def set_batch_size(net, size):
    net.batch_size = size


def load_network(model_full_path, device, extensions, batch=1, new_shape=None):
    from openvino.inference_engine import IENetwork, IEPlugin

    #  Read in Graph file (IR)
    net = IENetwork.from_ir(model=model_full_path + ".xml", weights=model_full_path + ".bin")

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    #  Plugin initialization for specified device and load extensions library if needed
    plugin = IEPlugin(device=device.upper())

    if 'extension library' in extensions[device]:
        plugin.add_cpu_extension(extensions[device]['extension library'])

    if device.upper() == 'MYRIAD':
        # TODO: set to true if you want to use multiple NCS devices
        # plugin.set_config({"VPU_FORCE_RESET": "YES"})
        exec_net = plugin.load(network=net)
    else:
        net.batch_size = batch

        exec_net = plugin.load(network=net)
        # exec_net = plugin.load(network=net, config={'DYN_BATCH_ENABLED': 'YES'})

    del net
    return plugin, exec_net, input_blob, out_blob
