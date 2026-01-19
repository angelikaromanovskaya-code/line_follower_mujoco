import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
from line_follower import LineFollowerCV
import config  # Константы из config.py

def get_camera_image(model, data, camera_name="camera_down", renderer=None):
    """Получаем изображение с камеры"""
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    
    if renderer is None:
        renderer = mujoco.Renderer(model, height=config.IMG_HEIGHT, width=config.IMG_WIDTH)
    
    renderer.update_scene(data, camera=camera_id)
    image = renderer.render()
    
    return image, renderer

def main():
    # Загружаем модель
    try:
        model = mujoco.MjModel.from_xml_path(config.MODEL_PATH)
    except Exception as e:
        print(f"Ошибка загрузки XML файла: {e}")
        print("Текущая папка:", os.getcwd())
        return
    
    data = mujoco.MjData(model)
    
    # Начальная позиция робота
    data.qpos[0] = 0.3    # x
    data.qpos[1] = 3.5   # y 
    data.qpos[2] = 0.1  # z
    
    # Инициализируем обработчик CV
    cv_processor = LineFollowerCV()
    
    # Создаем рендерер для камеры
    renderer = mujoco.Renderer(model, height=config.IMG_HEIGHT, width=config.IMG_WIDTH)
    
    # Состояние
    is_stopped = False
    no_line_counter = 0
    
    print("=" * 50)
    print("СИМУЛЯЦИЯ СЛЕДОВАНИЯ ПО ЛИНИИ С КАМЕРОЙ")
    print("=" * 50)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
       
        print("Стабилизация робота...")
        for i in range(100):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)
        
        print("Старт!")
        
        step = 0
        
        # Окно для отображения обработанного изображения
        cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera View", 400, 400)

        cv2.namedWindow("Camera View - Original", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera View - Original", 400, 400)
        
        while viewer.is_running():
            step += 1
            
            if not is_stopped:
                mujoco.mj_step(model, data)
            
            # Изображение с камеры
            camera_image, renderer = get_camera_image(model, data, "camera_down", renderer)

            if camera_image is not None:
              orig_display = cv2.resize(camera_image, (400, 400))
              orig_display = cv2.cvtColor(orig_display, cv2.COLOR_RGB2BGR)
              cv2.imshow("Camera View - Original", orig_display)
            
            # Обрабатка изображение
            left_val, center_val, right_val = cv_processor.process_image(camera_image)
            
            # Отображание обработанного изображения
            if hasattr(cv_processor, 'debug_image'):
                display_img = cv2.resize(cv_processor.debug_image, (400, 400), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Camera View", display_img)
            
            # Определение состояние
            center_on_line = center_val > config.LINE_THRESHOLD
            left_on_line = left_val > config.LINE_THRESHOLD
            right_on_line = right_val > config.LINE_THRESHOLD
            
            # Позиция робота
            robot_y = data.qpos[1]
            
            # Проверка выхода за пределы
            if robot_y < -3 and not is_stopped:
                print(f"\n[Шаг {step}] Робот проехал за пределы линии!")
                is_stopped = True
                data.ctrl[0] = 0.0
                data.ctrl[1] = 0.0
            
            if not is_stopped:
                # Логика управления на основе CV
                if center_on_line:
                    # Центр видит линию
                    data.ctrl[0] = config.BASE_SPEED
                    data.ctrl[1] = 0.0
                    no_line_counter = 0
                    
                elif left_on_line:
                    # Линия слева -> поворачиваем ВПРАВО
                    data.ctrl[0] = config.BASE_SPEED * 0.6
                    data.ctrl[1] = -config.TURN
                    no_line_counter = 0
                    
                elif right_on_line:
                    # Линия справа -> поворачиваем ВЛЕВО
                    data.ctrl[0] = config.BASE_SPEED * 0.6
                    data.ctrl[1] = config.TURN
                    no_line_counter = 0
                    
                else:
                    # Линия не обнаружена
                    no_line_counter += 1
                    
                    if no_line_counter > config.MAX_NO_LINE:
                        print(f"\n[Шаг {step}] ОСТАНОВКА!")
                        is_stopped = True
                        data.ctrl[0] = 0.0
                        data.ctrl[1] = 0.0
                    else:
                        # Поиск линии
                        data.ctrl[0] = 0.05
                        data.ctrl[1] = config.TURN * 0.8
                
                # Периодический вывод информации
                if step % 100 == 0:
                    print(f"[Шаг {step}] y={robot_y:.2f} | L={left_val:.2f} | C={center_val:.2f} | R={right_val:.2f}")
            
            viewer.sync()
            
            # Обработка клавиш OpenCV
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' или ESC
                break
        
        cv2.destroyAllWindows()