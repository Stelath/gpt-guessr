from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from getpass import getpass
import os
from selenium.webdriver.common.by import By
from PIL import Image
from selenium.webdriver.common.action_chains import ActionChains
import time    
import sys

def run_test():
        driver = webdriver.Chrome()

        url = sys.argv[1]
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'styles_root__3xbKq')))

        elem = driver.find_element(By.CLASS_NAME, "styles_root__3xbKq")
        driver.execute_script("""
        var element = arguments[0];
        element.parentNode.removeChild(element);
        """, elem)

        elem = driver.find_element(By.CLASS_NAME, "game-layout__top-hud")
        driver.execute_script("""
        var element = arguments[0];
        element.parentNode.removeChild(element);
        """, elem)

        elem = driver.find_element(By.CLASS_NAME, "status_inner__1eytg")
        driver.execute_script("""
        var element = arguments[0];
        element.parentNode.removeChild(element);
        """, elem)  

        elem = driver.find_element(By.CLASS_NAME, "gmnoprint")
        driver.execute_script("""
        var element = arguments[0];
        element.parentNode.removeChild(element);
        """, elem)

        elem = driver.find_element(By.CLASS_NAME, "gmnoprint")
        driver.execute_script("""
        var element = arguments[0];
        element.parentNode.removeChild(element);
        """, elem)

        elem = driver.find_element(By.CLASS_NAME, "gmnoprint")
        driver.execute_script("""
        var element = arguments[0];
        element.parentNode.removeChild(element);
        """, elem)

        elem = driver.find_element(By.CLASS_NAME, "game-layout__guess-map")
        driver.execute_script("""
        var element = arguments[0];
        element.parentNode.removeChild(element);
        """, elem)

        driver.save_screenshot("Image.png")
        h = driver.find_element(By.TAG_NAME, "main")
        ActionChains(driver)\
                .move_to_element(h)\
                .click_and_hold(h)\
                .move_by_offset(260, 0)\
                .release(h)\
                .move_to_element(h)\
                .click_and_hold(h)\
                .move_by_offset(260, 0)\
                .release(h)\
                .move_to_element(h)\
                .click_and_hold(h)\
                .move_by_offset(260, 0)\
                .release(h)\
                .perform()
        driver.save_screenshot("Image2.png")
        ActionChains(driver)\
                .move_to_element(h)\
                .click_and_hold(h)\
                .move_by_offset(260, 0)\
                .release(h)\
                .move_to_element(h)\
                .click_and_hold(h)\
                .move_by_offset(260, 0)\
                .release(h)\
                .move_to_element(h)\
                .click_and_hold(h)\
                .move_by_offset(260, 0)\
                .release(h)\
                .perform()
        driver.save_screenshot("Image3.png")

        #send the images to the AI and receive the location estimates

        os.remove("Image.png")
        os.remove("Image2.png")
        os.remove("Image3.png")

        coords = [-45.284, 28.645]
        print(coords)
        return coords

run_test()
