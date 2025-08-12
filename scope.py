#!/usr/bin/env python3
"""
Remote FEI Microscope Connector
Connects to FEI microscope at 192.168.0.1 via pyscope protocol
Bypasses all local import/compatibility issues
"""

import socket
import pickle
import time
import sys
from datetime import datetime

# Pyscope protocol constants
PYSCOPE_PORT = 55555
FEI_HOST = '128.186.1.176'
# FEI_HOST = '192.168.0.1'


class PyscopeData(object):
    """Base class for all data passed between client and server"""

    def __init__(self):
        self.login = None


class LoginRequest(PyscopeData):
    """Request login to a pyscope server"""

    def __init__(self, status):
        PyscopeData.__init__(self)
        self.status = status


class LoginResponse(PyscopeData):
    """Server response to client login request"""

    def __init__(self, status):
        PyscopeData.__init__(self)
        self.status = status


class CapabilityRequest(PyscopeData):
    """Request the set of capabilities provided by server"""
    pass


class CapabilityResponse(PyscopeData, dict):
    """Response with set of capabilities provided by server"""

    def __init__(self, initializer={}):
        PyscopeData.__init__(self)
        dict.__init__(self, initializer)


class InstrumentData(PyscopeData):
    """Base class for instrument specific data"""

    def __init__(self, instrument):
        PyscopeData.__init__(self)
        self.instrument = instrument


class GetRequest(list, InstrumentData):
    def __init__(self, instrument, sequence=[]):
        list.__init__(self, sequence)
        InstrumentData.__init__(self, instrument)


class GetResponse(dict, InstrumentData):
    def __init__(self, instrument, initializer={}):
        dict.__init__(self, initializer)
        InstrumentData.__init__(self, instrument)


class FEIRemoteClient:
    """Simple pyscope client for connecting to remote FEI microscope"""

    def __init__(self, host=FEI_HOST, port=PYSCOPE_PORT, login='fei_user'):
        self.host = host
        self.port = port
        self.login = login
        self.connected = False
        self.instruments = {}

    def connect(self):
        """Connect to remote pyscope server"""
        try:
            print(f"Connecting to {self.host}:{self.port}...")

            # Test basic connectivity first
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.host, self.port))
            sock.close()

            if result != 0:
                raise ConnectionError(f"Cannot reach {self.host}:{self.port}")

            # Login to pyscope server
            self._do_login()

            # Get available instruments
            self._get_capabilities()

            self.connected = True
            print(f"✓ Successfully connected to {self.host}")
            return True

        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def _create_socket(self):
        """Create socket connection for single request"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((self.host, self.port))
        return sock

    def _send_request(self, request):
        """Send request and get response"""
        request.login = self.login

        sock = self._create_socket()
        try:
            # Send request
            wfile = sock.makefile('wb')
            pickle.dump(request, wfile)
            wfile.flush()

            # Get response
            rfile = sock.makefile('rb')
            response = pickle.load(rfile)

            wfile.close()
            rfile.close()

            return response

        finally:
            sock.close()

    def _do_login(self):
        """Login to pyscope server"""
        request = LoginRequest('connected')
        response = self._send_request(request)

        if response.status != 'connected':
            raise Exception(f'Login failed: {response.status}')

    def _get_capabilities(self):
        """Get available instruments from server"""
        request = CapabilityRequest()
        response = self._send_request(request)

        self.instruments = response
        print(f"Found {len(self.instruments)} instruments:")

        for name, caps in self.instruments.items():
            inst_type = caps.get('type', 'Unknown')
            print(f"  - {name} ({inst_type})")

    def get_instrument_property(self, instrument_name, property_list):
        """Get properties from an instrument"""
        if not self.connected:
            raise Exception("Not connected to server")

        if instrument_name not in self.instruments:
            raise Exception(f"Instrument '{instrument_name}' not found")

        request = GetRequest(instrument_name, property_list)
        response = self._send_request(request)

        return response

    def get_fei_instruments(self):
        """Get list of FEI/TEM instruments"""
        fei_instruments = []

        for name, caps in self.instruments.items():
            inst_type = caps.get('type', '')

            # Check if it's a TEM or FEI instrument
            if (inst_type == 'TEM' or
                    any(keyword in name.lower() for keyword in ['fei', 'tecnai', 'titan', 'krios', 'talos'])):
                fei_instruments.append(name)

        return fei_instruments

    def get_stage_position(self, instrument_name):
        """Get stage position from FEI microscope"""
        try:
            response = self.get_instrument_property(instrument_name, ['StagePosition'])
            return response.get('StagePosition')
        except Exception as e:
            print(f"Error getting stage position: {e}")
            return None

    def monitor_stage_position(self, instrument_name, duration=10, interval=1):
        """Monitor stage position continuously"""
        print(f"\nMonitoring {instrument_name} stage position for {duration} seconds...")
        print("Press Ctrl+C to stop early")
        print("-" * 60)

        start_time = time.time()
        last_position = None

        try:
            while (time.time() - start_time) < duration:
                try:
                    position = self.get_stage_position(instrument_name)

                    if position:
                        elapsed = time.time() - start_time

                        # Check for movement
                        moved = ""
                        if last_position:
                            dx = abs(position.get('x', 0) - last_position.get('x', 0)) * 1e6
                            dy = abs(position.get('y', 0) - last_position.get('y', 0)) * 1e6
                            dz = abs(position.get('z', 0) - last_position.get('z', 0)) * 1e6
                            if dx > 0.1 or dy > 0.1 or dz > 0.1:
                                moved = " ⟵ MOVED"

                        print(f"[{elapsed:5.1f}s] "
                              f"X:{position.get('x', 0) * 1e6:8.2f}μm "
                              f"Y:{position.get('y', 0) * 1e6:8.2f}μm "
                              f"Z:{position.get('z', 0) * 1e6:8.2f}μm{moved}")

                        last_position = position.copy()
                    else:
                        elapsed = time.time() - start_time
                        print(f"[{elapsed:5.1f}s] No position data")

                    time.sleep(interval)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error during monitoring: {e}")
                    time.sleep(interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")


def print_stage_position(position, instrument_name):
    """Print stage position in readable format"""
    if not position:
        print("No position data available")
        return

    print(f"\n📍 {instrument_name} - CURRENT STAGE POSITION:")
    print("=" * 50)

    if 'x' in position and position['x'] is not None:
        print(f"X: {position['x'] * 1e6:10.2f} μm")
    if 'y' in position and position['y'] is not None:
        print(f"Y: {position['y'] * 1e6:10.2f} μm")
    if 'z' in position and position['z'] is not None:
        print(f"Z: {position['z'] * 1e6:10.2f} μm")
    if 'a' in position and position['a'] is not None:
        print(f"Alpha: {position['a'] * 180 / 3.14159:7.2f} °")
    if 'b' in position and position['b'] is not None:
        print(f"Beta:  {position['b'] * 180 / 3.14159:7.2f} °")


def test_all_instruments(client):
    """Test stage position reading on all FEI instruments"""
    fei_instruments = client.get_fei_instruments()

    if not fei_instruments:
        print("No FEI/TEM instruments found")
        return []

    working_instruments = []

    for instrument in fei_instruments:
        print(f"\nTesting {instrument}...")
        position = client.get_stage_position(instrument)

        if position:
            print(f"✓ Successfully read stage position from {instrument}")
            print_stage_position(position, instrument)
            working_instruments.append(instrument)
        else:
            print(f"✗ Could not read stage position from {instrument}")

    return working_instruments


def generate_code_examples(host, working_instruments):
    """Generate working code examples"""
    if not working_instruments:
        print("\nNo working instruments found - cannot generate examples")
        return

    instrument = working_instruments[0]

    print(f"\n" + "=" * 60)
    print("WORKING CODE EXAMPLES")
    print("=" * 60)

    print("🎯 Basic Remote FEI Connection:")
    print("-" * 40)
    print("```python")
    print("import socket")
    print("import pickle")
    print("")
    print("# Copy the FEIRemoteClient class from this script")
    print("# Then use it like this:")
    print("")
    print(f"client = FEIRemoteClient(host='{host}')")
    print("client.connect()")
    print("")
    print(f"# Get stage position from {instrument}")
    print(f"position = client.get_stage_position('{instrument}')")
    print("print(f'X: {position[\"x\"]*1e6:.2f} μm')")
    print("print(f'Y: {position[\"y\"]*1e6:.2f} μm')")
    print("print(f'Z: {position[\"z\"]*1e6:.2f} μm')")
    print("```")

    print(f"\n🔄 Continuous Monitoring:")
    print("-" * 40)
    print("```python")
    print("# Monitor stage position for 60 seconds")
    print(f"client.monitor_stage_position('{instrument}', duration=60)")
    print("```")


def main():
    """Main function"""
    print("🔬 Remote FEI Microscope Connector")
    print("=" * 60)
    print(f"Connecting to FEI microscope at {FEI_HOST}...")
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Create client and connect
        client = FEIRemoteClient(host=FEI_HOST)

        if not client.connect():
            print("❌ Failed to connect to remote FEI microscope")
            sys.exit(1)

        # Test all FEI instruments
        working_instruments = test_all_instruments(client)

        if working_instruments:
            print(f"\n🎉 SUCCESS!")
            print(f"✓ Connected to remote FEI microscope at {FEI_HOST}")
            print(f"✓ Found {len(working_instruments)} working FEI instruments:")
            for inst in working_instruments:
                print(f"   - {inst}")

            # Do continuous monitoring on first working instrument
            if working_instruments:
                print(f"\n🔄 Testing continuous monitoring...")
                client.monitor_stage_position(working_instruments[0], duration=5, interval=0.5)

            # Generate code examples
            generate_code_examples(FEI_HOST, working_instruments)

        else:
            print(f"\n⚠️ PARTIAL SUCCESS")
            print(f"✓ Connected to remote server at {FEI_HOST}")
            print(f"✗ No working FEI instruments found")
            print(f"\nAvailable instruments:")
            for name, caps in client.instruments.items():
                print(f"   - {name} ({caps.get('type', 'Unknown')})")

    except KeyboardInterrupt:
        print(f"\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nTest completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()