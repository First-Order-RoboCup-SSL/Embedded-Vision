# Embedded-Vision
Code for vision. Don't read me, read the code : P

I'm currently testing it and will delete the venv shortly.


# TODO
[] Check the performance at different lighting conditions
[] Add PD control -> center the ball
[] usb communication
[] 

# Libraries
<h2>Core Libraries</h2>
<pre><code>sudo apt install -y git
sudo apt install -y python3-opencv
sudo apt install -y python3-pip
pip install numba --break-system-packages</code></pre>

<h2>Firmware Config</h2>
<pre><code>sudo nano /boot/firmware/config.txt</code></pre>
<p>Set <code>camera_auto_detect=1</code> to <code>0</code></p>
<p>Add underneath:</p>
<pre><code>dtoverlay=imx219,cam1</code></pre>

<h2>Libcamera Dependencies</h2>
<pre><code>sudo apt install -y build-essential
sudo apt install -y libboost-dev
sudo apt install -y libgnutls28-dev openssl libtiff-dev pybind11-dev
sudo apt install -y qtbase5-dev libqt5core5a libqt5widgets5t64
sudo apt install -y meson cmake
sudo apt install -y python3-yaml python3-ply
sudo apt install -y libglib2.0-dev libgstreamer-plugins-base1.0-dev</code></pre>

<h2>Building Libcamera</h2>
<pre><code>cd
git clone https://github.com/raspberrypi/libcamera.git
cd libcamera
meson setup build --buildtype=release -Dgstreamer=enabled -Dpycamera=enabled
sudo ninja -C build install</code></pre>

<h2>GStreamer</h2>
<pre><code>sudo apt install gstreamer1.0-tools
sudo ldconfig
rm ~/.cache/gstreamer-1.0/registry.*.bin
echo 'export GST_PLUGIN_PATH=/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PYTHONPATH=/usr/local/lib/python3/dist-packages:$PYTHONPATH' >> ~/.bashrc
gst-inspect-1.0 libcamerasrc</code></pre>
<p>Test command above.</p>

<h2>User Permission</h2>
<pre><code>sudo usermod -aG video $USER
sudo reboot</code></pre>

<h2>Camera Test</h2>
<pre><code>gst-launch-1.0 libcamerasrc ! fakesink</code></pre>
