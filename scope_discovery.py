#!/usr/bin/env python3
"""
FEI Microscope Connection Tester
Automatically discovers and tests FEI microscope connections using multiple methods
"""

import sys
import os
import time
import socket
import traceback
from datetime import datetime


def test_imports():
    """Test if all required modules can be imported"""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)

    required_modules = [
        'pyscope',
        'pyscope.fei',
        'pyscope.remote',
        'pyscope.config',
        'pyami.moduleconfig',
        'leginon.leginondata'
    ]

    available_modules = {}

    for module in required_modules:
        try:
            exec(f"import {module}")
            available_modules[module] = True
            print(f"✓ {module}")
        except ImportError as e:
            available_modules[module] = False
            print(f"✗ {module} - {e}")
        except Exception as e:
            available_modules[module] = False
            print(f"? {module} - Unexpected error: {e}")

    return available_modules


def discover_network_hosts():
    """Discover potential microscope computers on the network"""
    print("\n" + "=" * 60)
    print("NETWORK DISCOVERY")
    print("=" * 60)

    # Get local network info
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Local hostname: {hostname}")
    print(f"Local IP: {local_ip}")

    # Common microscope computer names
    potential_hosts = [
        'localhost',
        '127.0.0.1',
        hostname,
        'tem-pc',
        'tem-pc-01',
        'microscope-pc',
        'fei-pc',
        'tecnai-pc',
        'titan-pc',
        'krios-pc',
        f'{hostname}-tem',
        f'{hostname}-microscope'
    ]

    # Add IP range scan for local subnet
    if '.' in local_ip:
        ip_parts = local_ip.split('.')
        if len(ip_parts) == 4:
            base_ip = '.'.join(ip_parts[:3])
            # Add common IP addresses in range
            for i in [1, 10, 50, 100, 101, 102, 200, 254]:
                potential_hosts.append(f"{base_ip}.{i}")

    print(f"\nScanning {len(potential_hosts)} potential hosts...")

    available_hosts = []
    for host in potential_hosts:
        try:
            # Quick connectivity test
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)  # 1 second timeout
            result = sock.connect_ex((host, 55555))  # pyscope default port
            sock.close()

            if result == 0:
                available_hosts.append(host)
                print(f"✓ {host}:55555 - pyscope server detected")
            else:
                print(f"  {host} - no pyscope server")

        except Exception as e:
            print(f"  {host} - connection error")

    return available_hosts


def test_database_discovery():
    """Test database-based instrument discovery"""
    print("\n" + "=" * 60)
    print("DATABASE DISCOVERY")
    print("=" * 60)

    try:
        import leginon.leginondata as leginondata

        # Query for all instruments
        query = leginondata.InstrumentData()
        instruments = query.query()

        fei_instruments = []
        all_instruments = []

        print(f"Found {len(instruments)} instruments in database:")

        for inst in instruments:
            name = inst.get('name', 'Unknown')
            hostname = inst.get('hostname', 'Unknown')
            inst_type = inst.get('type', 'Unknown')

            all_instruments.append({
                'name': name,
                'hostname': hostname,
                'type': inst_type
            })

            print(f"  - {name} on {hostname} (type: {inst_type})")

            # Check if it's FEI-related
            name_lower = name.lower()
            if any(keyword in name_lower for keyword in ['fei', 'tecnai', 'titan', 'krios', 'talos', 'polara']):
                fei_instruments.append({
                    'name': name,
                    'hostname': hostname,
                    'type': inst_type
                })
                print(f"    → FEI instrument detected!")

        return fei_instruments, all_instruments

    except Exception as e:
        print(f"Database discovery failed: {e}")
        print("This is normal if database is not configured")
        return [], []


def test_config_discovery():
    """Test configuration file discovery"""
    print("\n" + "=" * 60)
    print("CONFIGURATION DISCOVERY")
    print("=" * 60)

    try:
        from pyscope import config

        # Parse instruments.cfg
        configured = config.getConfigured()

        fei_configs = []

        print(f"Found {len(configured)} configured instruments:")

        for name, cls in configured.items():
            class_name = cls.__name__ if hasattr(cls, '__name__') else str(cls)
            print(f"  - {name}: {class_name}")

            # Check if it's FEI-related
            if any(keyword in class_name.lower() for keyword in ['fei', 'tecnai']):
                fei_configs.append({
                    'name': name,
                    'class': class_name,
                    'hostname': socket.gethostname()  # Assume local
                })
                print(f"    → FEI instrument detected!")

        return fei_configs

    except Exception as e:
        print(f"Configuration discovery failed: {e}")
        traceback.print_exc()
        return []


def test_direct_fei_connection():
    """Test direct FEI connection (local)"""
    print("\n" + "=" * 60)
    print("DIRECT FEI CONNECTION TEST")
    print("=" * 60)

    connection_results = []

    # Try different FEI connection methods
    fei_classes = [
        ('pyscope.fei', 'Tecnai'),
        ('pyscope.fei', 'FEI'),
        ('pyscope.tecnai', 'Tecnai'),
    ]

    for module_name, class_name in fei_classes:
        try:
            print(f"\nTesting {module_name}.{class_name}...")

            # Import the module
            module = __import__(module_name, fromlist=[class_name])
            fei_class = getattr(module, class_name)

            # Try to instantiate
            print(f"  Instantiating {class_name}...")
            fei_instance = fei_class()

            print(f"  ✓ Successfully created {class_name} instance")

            # Test basic functionality
            print(f"  Testing getStagePosition()...")
            position = fei_instance.getStagePosition()

            if position:
                print(
                    f"  ✓ Stage position: X={position.get('x', 0) * 1e6:.2f}μm, Y={position.get('y', 0) * 1e6:.2f}μm, Z={position.get('z', 0) * 1e6:.2f}μm")
                connection_results.append({
                    'method': 'direct',
                    'class': f"{module_name}.{class_name}",
                    'status': 'success',
                    'position': position,
                    'instance': fei_instance
                })
            else:
                print(f"  ⚠ Connected but no position data")
                connection_results.append({
                    'method': 'direct',
                    'class': f"{module_name}.{class_name}",
                    'status': 'partial',
                    'position': None,
                    'instance': fei_instance
                })

        except ImportError as e:
            print(f"  ✗ Import failed: {e}")
        except Exception as e:
            print(f"  ✗ Connection failed: {e}")

    return connection_results


def test_remote_connections(hosts, instruments):
    """Test remote connections to discovered hosts"""
    print("\n" + "=" * 60)
    print("REMOTE CONNECTION TEST")
    print("=" * 60)

    connection_results = []

    try:
        from pyscope.remote import Client

        for host in hosts:
            print(f"\nTesting connection to {host}...")

            try:
                # Create client connection
                client = Client(
                    login='test_user',
                    status='connected',
                    host=host,
                    port=55555
                )

                print(f"  ✓ Connected to {host}")

                # Get capabilities
                capabilities = client.getCapabilities()
                print(f"  Found {len(capabilities)} instruments:")

                for inst_name, caps in capabilities.items():
                    inst_type = caps.get('type', 'Unknown')
                    print(f"    - {inst_name} ({inst_type})")

                    # Test if it's a TEM and try to get stage position
                    if inst_type == 'TEM':
                        try:
                            print(f"      Testing stage position...")
                            result = client.get(inst_name, ['StagePosition'])
                            position = result.get('StagePosition')

                            if position:
                                print(
                                    f"      ✓ X={position.get('x', 0) * 1e6:.2f}μm, Y={position.get('y', 0) * 1e6:.2f}μm, Z={position.get('z', 0) * 1e6:.2f}μm")
                                connection_results.append({
                                    'method': 'remote',
                                    'host': host,
                                    'instrument': inst_name,
                                    'status': 'success',
                                    'position': position,
                                    'client': client
                                })
                            else:
                                print(f"      ⚠ No position data")

                        except Exception as e:
                            print(f"      ✗ Stage position failed: {e}")

                # Try to logout cleanly
                try:
                    client.logout()
                except:
                    pass

            except Exception as e:
                print(f"  ✗ Connection to {host} failed: {e}")

    except ImportError as e:
        print(f"Remote connection module not available: {e}")

    return connection_results


def test_continuous_monitoring(connection_results):
    """Test continuous stage position monitoring"""
    print("\n" + "=" * 60)
    print("CONTINUOUS MONITORING TEST")
    print("=" * 60)

    if not connection_results:
        print("No successful connections available for monitoring")
        return

    # Use the first successful connection
    conn = connection_results[0]

    print(f"Using connection: {conn['method']} - {conn.get('class', conn.get('instrument', 'unknown'))}")
    print("Monitoring stage position for 10 seconds (press Ctrl+C to stop early)...")

    try:
        start_time = time.time()

        while (time.time() - start_time) < 10:
            try:
                if conn['method'] == 'direct':
                    position = conn['instance'].getStagePosition()
                elif conn['method'] == 'remote':
                    result = conn['client'].get(conn['instrument'], ['StagePosition'])
                    position = result.get('StagePosition')
                else:
                    break

                if position:
                    elapsed = time.time() - start_time
                    print(
                        f"[{elapsed:5.1f}s] X:{position.get('x', 0) * 1e6:8.2f}μm Y:{position.get('y', 0) * 1e6:8.2f}μm Z:{position.get('z', 0) * 1e6:8.2f}μm")
                else:
                    print(f"[{elapsed:5.1f}s] No position data")

                time.sleep(0.5)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                break

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")


def generate_summary_report(import_results, network_hosts, db_instruments, config_instruments, connection_results):
    """Generate a comprehensive summary report"""
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Test completed at: {timestamp}")

    print(f"\n📦 Module Availability:")
    for module, available in import_results.items():
        status = "✓" if available else "✗"
        print(f"  {status} {module}")

    print(f"\n🌐 Network Discovery:")
    print(f"  Found {len(network_hosts)} hosts with pyscope servers:")
    for host in network_hosts:
        print(f"    - {host}")

    print(f"\n💾 Database Discovery:")
    print(f"  Found {len(db_instruments)} FEI instruments in database:")
    for inst in db_instruments:
        print(f"    - {inst['name']} on {inst['hostname']}")

    print(f"\n⚙️ Configuration Discovery:")
    print(f"  Found {len(config_instruments)} FEI instruments in config:")
    for inst in config_instruments:
        print(f"    - {inst['name']} ({inst['class']})")

    print(f"\n🔗 Successful Connections:")
    successful_connections = [c for c in connection_results if c['status'] == 'success']
    print(f"  Found {len(successful_connections)} working connections:")

    for conn in successful_connections:
        if conn['method'] == 'direct':
            print(f"    ✓ Direct: {conn['class']}")
        elif conn['method'] == 'remote':
            print(f"    ✓ Remote: {conn['instrument']} on {conn['host']}")

        if conn['position']:
            pos = conn['position']
            print(
                f"      Stage: X={pos.get('x', 0) * 1e6:.2f}μm Y={pos.get('y', 0) * 1e6:.2f}μm Z={pos.get('z', 0) * 1e6:.2f}μm")

    print(f"\n🎯 Recommendations:")
    if successful_connections:
        conn = successful_connections[0]
        print(f"  ✓ FEI microscope found and working!")

        if conn['method'] == 'direct':
            print(f"  📝 Use this code to connect:")
            print(f"     from {conn['class'].split('.')[0]} import {conn['class'].split('.')[1]}")
            print(f"     tem = {conn['class'].split('.')[1]}()")
            print(f"     position = tem.getStagePosition()")

        elif conn['method'] == 'remote':
            print(f"  📝 Use this code to connect:")
            print(f"     from pyscope.remote import Client")
            print(f"     client = Client('user', 'connected', '{conn['host']}')")
            print(f"     position = client.get('{conn['instrument']}', ['StagePosition'])")
    else:
        print(f"  ⚠️ No working FEI connections found")
        print(f"  💡 Check:")
        print(f"     - FEI software is running")
        print(f"     - pyscope server is started")
        print(f"     - Network connectivity")
        print(f"     - Configuration files")


def main():
    """Main test function"""
    print("FEI Microscope Connection Tester")
    print("Discovering and testing all available FEI microscope connections...")
    print("This may take a few minutes...")

    try:
        # Test 1: Check imports
        import_results = test_imports()

        # Test 2: Network discovery
        network_hosts = discover_network_hosts()

        # Test 3: Database discovery
        db_instruments, all_db_instruments = test_database_discovery()

        # Test 4: Configuration discovery
        config_instruments = test_config_discovery()

        # Test 5: Direct connections
        direct_results = test_direct_fei_connection()

        # Test 6: Remote connections
        remote_results = test_remote_connections(network_hosts, db_instruments)

        # Combine all connection results
        all_connections = direct_results + remote_results

        # Test 7: Continuous monitoring
        if all_connections:
            test_continuous_monitoring(all_connections)

        # Generate final report
        generate_summary_report(
            import_results,
            network_hosts,
            db_instruments,
            config_instruments,
            all_connections
        )

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        traceback.print_exc()

    print("\nTest complete!")


if __name__ == "__main__":
    main()