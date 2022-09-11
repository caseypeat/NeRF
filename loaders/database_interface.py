from camera_geometry.scan.scan import Scan


if __name__ == '__main__':

    Scan.load("remote://row/178/50")

    # auth_client = AuthenticatedClient("http://csse-maaratech1.canterbury.ac.nz/", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJjcGVfcmVzZWFyY2hlciIsInJvbGVzIjpbInJlc2VhcmNoZXIiXSwiZXhwIjoxNjYyNTgzOTI0fQ.itH0ICv2p6IkQFck3fabm5A8kUlVwdblaCxKHuB3u3A")
    # a = AuthenticatedClient("http://localhost:5000/", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJjcGVfcmVzZWFyY2hlciIsInJvbGVzIjpbInJlc2VhcmNoZXIiXSwiZXhwIjoxNjYyNTgzOTI0fQ.itH0ICv2p6IkQFck3fabm5A8kUlVwdblaCxKHuB3u3A")

    # print(a)

    # asyncio.run(scan_tools.remote.query.scan_query(a, [384]))
    # asyncio.run(scan_tools.remote.query.scan_query(auth_client, [1, 2]))
    # asyncio.run(scan_tools.remote.query.scan_query(a, )
    # print(scan_tools.remote.query.scan_query(a, [1]))

    # cfg = login.load_config()

    # token_file = login.token_file(cfg)

    # print(cfg)
    # print(token_file)

    # Struct()

    # login.login(cfg, {"user":"cpe_researcher", "pass":"7Xy@3X3b^&U2"})
    # login.login(cfg)

    # login()
