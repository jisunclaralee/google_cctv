"""
CCTV 용의자 식별 시스템 - Flask 백엔드 서버
InsightFace (RetinaFace + ArcFace) 모델 통합
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from flasgger import Swagger
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
import json
from datetime import datetime

# 모델 관련 imports - TODO: 실제 구현 필요
# from models.face_detector import FaceDetector
# from models.face_recognizer import FaceRecognizer  
# from models.embedding_db import EmbeddingDatabase

# API 엔드포인트
from api.upload import upload_bp
from api.detect import detect_bp
from api.suspects import suspects_bp

app = Flask(__name__)
CORS(app)

# Swagger UI 설정
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs"
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "CCTV 용의자 식별 시스템 API",
        "description": "InsightFace (RetinaFace + ArcFace) 기반 얼굴 인식 API",
        "version": "1.0.0",
        "contact": {
            "name": "API Support"
        }
    },
    "basePath": "/",
    "schemes": ["http", "https"],
    "tags": [
        {
            "name": "Status",
            "description": "시스템 상태 확인"
        },
        {
            "name": "Detection",
            "description": "얼굴 감지 및 인식"
        },
        {
            "name": "Suspects",
            "description": "용의자 관리"
        },
        {
            "name": "Upload",
            "description": "파일 업로드"
        }
    ]
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

# 설정
app.config['UPLOAD_FOLDER'] = 'data/videos'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB 제한
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'wmv'}

# 전역 변수 - 모델 인스턴스
face_detector = None
face_recognizer = None
embedding_db = None

def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def initialize_models():
    """AI 모델들 초기화"""
    # ===============================================================================
    # **중요: 실제 AI 모델 설치 및 초기화 필요**
    # ===============================================================================
    # TODO: InsightFace 라이브러리 설치 필요: pip install insightface
    # TODO: 모델 파일 자동 다운로드 및 캐싱 구현
    # TODO: GPU/CPU 환경 자동 감지 및 최적화
    # ===============================================================================
    global face_detector, face_recognizer, embedding_db
    
    print("🔧 AI 모델 초기화 중...")
    
    try:
        # TODO: 실제 AI 모델 구현 필요
        # 1. 얼굴 검출 모델 (RetinaFace) 로드
        # face_detector = FaceDetector()
        print("⚠️ RetinaFace 얼굴 검출 모델 - 미구현")
        
        # 2. 얼굴 인식 모델 (ArcFace) 로드  
        # face_recognizer = FaceRecognizer()
        print("⚠️ ArcFace 얼굴 인식 모델 - 미구현")
        
        # 3. 임베딩 데이터베이스 초기화
        # embedding_db = EmbeddingDatabase()
        print("⚠️ 임베딩 데이터베이스 - 미구현")
        
        # 4. 기본 용의자 데이터 로드
        # embedding_db.load_default_suspects()
        print("⚠️ 기본 용의자 데이터 로드 - 미구현")
        
        return True  # 개발 모드에서는 True 반환
        
    except Exception as e:
        print(f"❌ 모델 초기화 실패: {str(e)}")
        return False

@app.route('/')
def index():
    """메인 페이지 - HTML 파일 서빙"""
    try:
        with open('cctv_suspect_identification.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <h1>CCTV 용의자 식별 시스템</h1>
        <p>HTML 파일을 찾을 수 없습니다. cctv_suspect_identification.html 파일을 확인해주세요.</p>
        <p><a href="/api/status">시스템 상태 확인</a></p>
        """

@app.route('/api/status')
def status():
    """시스템 상태 확인 API
    ---
    tags:
      - Status
    responses:
      200:
        description: 시스템 상태 정보
        schema:
          type: object
          properties:
            timestamp:
              type: string
              example: "2025-11-18T03:20:00"
            models:
              type: object
              properties:
                face_detector:
                  type: boolean
                face_recognizer:
                  type: boolean
                embedding_db:
                  type: boolean
            database:
              type: object
              properties:
                suspects_count:
                  type: integer
                embeddings_loaded:
                  type: boolean
            system:
              type: object
              properties:
                opencv_version:
                  type: string
                upload_folder:
                  type: string
                max_file_size_mb:
                  type: integer
    """
    global face_detector, face_recognizer, embedding_db
    
    status_info = {
        'timestamp': datetime.now().isoformat(),
        'models': {
            'face_detector': face_detector is not None,
            'face_recognizer': face_recognizer is not None,
            'embedding_db': embedding_db is not None
        },
        'database': {
            'suspects_count': embedding_db.get_suspects_count() if embedding_db else 0,
            'embeddings_loaded': embedding_db.is_loaded() if embedding_db else False
        },
        'system': {
            'opencv_version': cv2.__version__,
            'upload_folder': app.config['UPLOAD_FOLDER'],
            'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] // (1024*1024)
        }
    }
    
    return jsonify(status_info)

@app.route('/api/detect_frame', methods=['POST'])
def detect_frame():
    """단일 프레임에서 얼굴 감지 및 인식
    ---
    tags:
      - Detection
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - image
          properties:
            image:
              type: string
              description: Base64 인코딩된 이미지 데이터 (data:image/jpeg;base64,... 형식)
            target_suspect_id:
              type: string
              description: 타겟 용의자 ID
              default: "1"
    responses:
      200:
        description: 감지 결과
        schema:
          type: object
          properties:
            success:
              type: boolean
            detections:
              type: array
              items:
                type: object
                properties:
                  bbox:
                    type: array
                    items:
                      type: number
                  confidence:
                    type: number
                  match:
                    type: object
                  timestamp:
                    type: string
            frame_info:
              type: object
              properties:
                width:
                  type: integer
                height:
                  type: integer
                faces_detected:
                  type: integer
      400:
        description: 잘못된 요청
      500:
        description: 서버 오류
    """
    # ===============================================================================
    # **중요: 실제 얼굴 인식 파이프라인 구현 필요**
    # ===============================================================================
    # TODO: Base64 이미지 디코딩 및 전처리 구현
    # TODO: RetinaFace 얼굴 검출 연동
    # TODO: ArcFace 특징 추출 연동
    # TODO: 실시간 매칭 및 임계값 설정
    # TODO: 검출 결과 로깅 및 알림 시스템
    # ===============================================================================
    global face_detector, face_recognizer, embedding_db
    
    try:
        # Base64 이미지 데이터 받기
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # Base64 디코딩
        image_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
            
        # 선택된 용의자 ID
        target_suspect_id = data.get('target_suspect_id', '1')
        
        # 1. 얼굴 검출 (RetinaFace)
        faces = face_detector.detect_faces(frame)
        
        results = []
        for face in faces:
            bbox, landmarks, confidence = face
            
            # 2. 얼굴 영역 추출
            x1, y1, x2, y2 = bbox.astype(int)
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                continue
                
            # 3. 얼굴 특징 추출 (ArcFace)
            embedding = face_recognizer.extract_embedding(face_roi)
            
            # 4. 데이터베이스와 매칭
            match_result = embedding_db.match_embedding(embedding, target_suspect_id)
            
            # 5. 결과 저장
            result = {
                'bbox': bbox.tolist(),
                'confidence': float(confidence),
                'match': match_result,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
        return jsonify({
            'success': True,
            'detections': results,
            'frame_info': {
                'width': frame.shape[1],
                'height': frame.shape[0],
                'faces_detected': len(results)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/suspects')
def get_suspects():
    """등록된 용의자 목록 반환
    ---
    tags:
      - Suspects
    responses:
      200:
        description: 용의자 목록
        schema:
          type: object
          properties:
            success:
              type: boolean
            suspects:
              type: array
              items:
                type: object
      500:
        description: 서버 오류
    """
    global embedding_db
    
    try:
        suspects = embedding_db.get_all_suspects()
        return jsonify({
            'success': True,
            'suspects': suspects
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/add_suspect', methods=['POST'])  
def add_suspect():
    """새로운 용의자 추가
    ---
    tags:
      - Suspects
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: name
        type: string
        required: true
        description: 용의자 이름
      - in: formData
        name: image
        type: file
        required: true
        description: 용의자 얼굴 이미지
      - in: formData
        name: criminal_record
        type: string
        description: 범죄 이력 (쉼표로 구분)
      - in: formData
        name: risk_level
        type: string
        enum: [low, medium, high]
        default: medium
        description: 위험도 레벨
    responses:
      200:
        description: 용의자 등록 성공
        schema:
          type: object
          properties:
            success:
              type: boolean
            suspect_id:
              type: string
            message:
              type: string
      400:
        description: 잘못된 요청
      500:
        description: 서버 오류
    """
    global face_detector, face_recognizer, embedding_db
    
    try:
        # 폼 데이터 받기
        name = request.form.get('name')
        criminal_record = request.form.get('criminal_record', '')
        risk_level = request.form.get('risk_level', 'medium')
        
        # 이미지 파일 받기
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
            
        # 이미지 처리
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 얼굴 검출 및 임베딩 추출
        faces = face_detector.detect_faces(image)
        if len(faces) == 0:
            return jsonify({'error': 'No face detected in image'}), 400
            
        # 첫 번째 얼굴 사용
        bbox, landmarks, confidence = faces[0]
        x1, y1, x2, y2 = bbox.astype(int)
        face_roi = image[y1:y2, x1:x2]
        
        # 임베딩 추출
        embedding = face_recognizer.extract_embedding(face_roi)
        
        # 데이터베이스에 저장
        suspect_id = embedding_db.add_suspect(
            name=name,
            embedding=embedding,
            criminal_record=criminal_record.split(',') if criminal_record else [],
            risk_level=risk_level
        )
        
        return jsonify({
            'success': True,
            'suspect_id': suspect_id,
            'message': f'용의자 {name}이 성공적으로 등록되었습니다.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Blueprint 등록
app.register_blueprint(upload_bp, url_prefix='/api')
app.register_blueprint(detect_bp, url_prefix='/api') 
app.register_blueprint(suspects_bp, url_prefix='/api')

@app.errorhandler(413)
def too_large(e):
    """파일 크기 초과 에러 핸들러"""
    return jsonify({
        'error': 'File too large. Maximum size is 100MB.'
    }), 413

if __name__ == '__main__':
    # 디렉터리 생성
    os.makedirs('data/videos', exist_ok=True)
    os.makedirs('data/suspects', exist_ok=True)
    os.makedirs('data/embeddings', exist_ok=True)
    
    # 모델 초기화
    if initialize_models():
        print("🚀 서버 시작 중...")
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
    else:
        print("❌ 모델 초기화 실패. 서버를 시작할 수 없습니다.")