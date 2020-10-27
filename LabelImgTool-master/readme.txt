< 실행 방법 >
    - 실행환경 : python 2.7.x

    - 우선 모든 source를 받아 적당한 폴더에 설치

    -  모든 라이브러리를 설치
        sudo apt-get install pyqt4-dev-tools
        sudo apt-get install python-opencv
        pip install lxml
        pip install qdarkstyle

    - 그럼에도 부족한 라이브러리가 있는데, 오류를 보고 검색을 해 본 후 해당 라이브러리를 설치하면 구동이 됐으니 그렇게 하면 될 듯 함. 아래 라이브러리 이외에도 필요한 것이 있으면 오류 메세지 확인 후 진행하면 됨
        pip install requests
        pip install pillow
        sudo apt install libcanberra-gtk-module libcanberra-gtk3-module

    - 실행은 다음과 같이 진행
        ./labelImg.py


< 주의할 점 >
* 이 프로그램은 open source를 받아 필요에 맞게 수정하여 사용 중에 있는 프로그램임
* 따라서 우리의 목적과 다른 수많은 기능과 버튼이 존재하고, 이들을 잘못 사용했을 때 무슨 문제가 발생할지 알 수 없으므로 건드리지 않는 것을 추천함
* 프로그램을 실행했을 때 왼쪽 위에 open 버튼을 눌러 이미지를 선택하면 메인 윈도우에 그림이 뜨고, 만약 마스크 파일이 있으면 이를 함께 읽어 마스크 레이어도 함께 올림 : labelImg.py / class MainWindow / openFile() 참고
* 여기에 왼쪽 마우스 버튼을 눌러 annotation을 진행. 단, 프로그램은 1단계, 2단계 annotation을 할 수 있도록 돼 있는데, 지금은 2단계만 사용하므로 반드시 '2' key를 눌러 2단계로 전환한 후 annotation을 진행해야 함
* 저장은 ctrl+s 로 하고 저장하면 동일파일이름.msk 파일이 동일폴더에 생김
* 기타 자세한 기능은 labelImg.py / class MainWindow / newOnkeyPressEvent() 를 참조하면 됨 : 이미지 크기 변환, 펜의 두께, 지우개 모드 토글, image panning(가운데 마우스 버튼 클릭 후 마우스 이동) 등의 기능이 있고, 키 배열이 불편하면 여기서 수정할 수 있음
* 많은 부분이 미완성으로 남아 있고 꼭 필요한 기능이 구동됨을 확인한 후 개발을 중지했기 때문에 사용에 주의가 필요하고 코드 분석이 어려울 수 있음
* /lib/maskInfoManager.py : 여기에 마스크 정보를 비트맵으로 변환해 저장하고 불러오는 모듈이 있으므로 이를 참고해 .msk 파일 형식을 분석할 수 있고 필요한 정보를 추출할 수 있음. 현재 마스크 레이어는 2개를 지원하고 이 중 두 번째만 사용하도록 annotation을 하고 있으니 두 번째 레이어를 사용해야 함
