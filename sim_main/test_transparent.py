import trollius
from trollius import From

import pygazebo
import pygazebo.msg.visual_pb2

@trollius.coroutine
def publish_loop():
    manager = yield From(pygazebo.connect(('172.16.0.1', 33459)))

    publisher = yield From(
        manager.advertise('/gazebo/megaweb/visual',
                          'gazebo.msgs.Visual'))

    message = pygazebo.msg.visual_pb2.Visual()
    message.name = 'tape3_0::link::Visual'
    # message.axis = 0
    message.transparency = 1.0

    while True:
        yield From(publisher.publish(message))
        yield From(trollius.sleep(1.0))

loop = trollius.get_event_loop()
loop.run_until_complete(publish_loop())