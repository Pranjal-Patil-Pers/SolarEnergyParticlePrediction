from socket import *
import struct
import time
import select
import os

ICMP_ECHO_REQUEST = 8
MAX_HOPS = 30
TIMEOUT = 2.0
TRIES = 2


# Basic checksum function used for ICMP packets
def checksum(data):
    c = 0
    count = 0
    length = len(data)
    while count < length - 1:
        val = data[count + 1] * 256 + data[count]
        c += val
        c &= 0xffffffff
        count += 2

    if length % 2:
        c += data[-1]
        c &= 0xffffffff

    c = (c >> 16) + (c & 0xffff)
    c += c >> 16
    answer = ~c & 0xffff
    return answer >> 8 | ((answer << 8) & 0xff00)


# Builds a simple ICMP echo request with timestamp
def build_packet():
    my_id = os.getpid() & 0xFFFF
    header = struct.pack("bbHHh", ICMP_ECHO_REQUEST, 0, 0, my_id, 1)
    data = struct.pack("d", time.time())

    cs = checksum(header + data)
    header = struct.pack("bbHHh", ICMP_ECHO_REQUEST, 0, cs, my_id, 1)
    return header + data


def get_route(hostname):
    for ttl in range(1, MAX_HOPS + 1):
        for _ in range(TRIES):
            try:
                dest_addr = gethostbyname(hostname)
            except:
                print("Could not resolve hostname.")
                return

            try:
                sock = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP)
            except PermissionError:
                print("You need to run this program with admin/root privileges.")
                return

            sock.setsockopt(IPPROTO_IP, IP_TTL, struct.pack("I", ttl))
            sock.settimeout(TIMEOUT)

            packet = build_packet()
            start_time = time.time()

            try:
                sock.sendto(packet, (hostname, 0))
                ready = select.select([sock], [], [], TIMEOUT)

                if ready[0] == []:
                    print(f"{ttl:2}   *   *   *   Request timed out")
                    continue

                recv_packet, addr = sock.recvfrom(1024)
                end_time = time.time()

            except timeout:
                continue
            finally:
                sock.close()

            # ICMP header is after the IP header (20 bytes)
            icmp_header = recv_packet[20:28]
            icmp_type, code, chk, pid, seq = struct.unpack("bbHHh", icmp_header)

            rtt = (end_time - start_time) * 1000

            if icmp_type == 11:   # Time Exceeded (normal hop)
                print(f"{ttl:2}   rtt={rtt:.0f} ms   {addr[0]}")

            elif icmp_type == 3:  # Destination unreachable
                print(f"{ttl:2}   rtt={rtt:.0f} ms   {addr[0]}")

            elif icmp_type == 0:  # Echo Reply (destination reached)
                print(f"{ttl:2}   rtt={rtt:.0f} ms   {addr[0]}")
                return

            else:
                print(f"{ttl:2}   Unexpected ICMP type: {icmp_type}")
                break


# Run traceroute
get_route("208.67.220.220")

