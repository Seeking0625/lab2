"""Optional real-Chrome UI smoke test. Uses isolated synthetic fixtures, never course labels.
Run with playwright installed (Chrome must be available).
"""
import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from werkzeug.serving import make_server
from playwright.sync_api import sync_playwright

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app
from utils.classifier import write_json


def rectangle(x,y,w,h,cls):
    return dict(id=0, **{'class':cls},bbox=[x,y,w,h],area=w*h,score=.99,
                polygons=[[[x,y],[x+w-1,y],[x+w-1,y+h-1],[x,y+h-1]]])


def main():
    with tempfile.TemporaryDirectory(prefix='lab2-ui-') as tmp:
        root=Path(tmp); images=root/'images'; results=root/'results'
        images.mkdir(); results.mkdir()
        app.app.config['REFERENCE_MODEL_PATH']=None
        app.app.config['ARCHIVE_IMAGES_ENABLED']=False
        app.IMAGE_DIR=str(images); app.RESULT_DIR=str(results); app.CLASS_FILE=str(root/'classes.json')
        shutil.copy(Path(app.BASE)/'classes.json',app.CLASS_FILE)
        for n in range(6):
            image=np.full((128,192,3),n,np.uint8)
            image[10:50,10:60]=[10,10,220+n]; image[60:110,90:160]=[10,220+n,10]
            cv2.imwrite(str(images/f'test{n}.png'),image)
            write_json(results/f'test{n}.json',dict(image=f'test{n}.png',width=192,height=128,
                instances=[dict(rectangle(10,10,50,40,'airplane 飞机'),id=0),
                           dict(rectangle(90,60,70,50,'building 建筑'),id=1)]))
        cv2.imwrite(str(images/'empty.png'),np.zeros((128,192,3),np.uint8))
        server=make_server('127.0.0.1',0,app.app,threaded=True)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        errors=[]; dialogs=[]
        try:
            with sync_playwright() as p:
                browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,
                    args=['--no-sandbox','--disable-dev-shm-usage'])
                page=browser.new_page(viewport={'width':1600,'height':1000})
                page.on('pageerror',lambda e:errors.append(str(e)))
                def dialog(d):
                    dialogs.append(d.message); d.accept()
                page.on('dialog',dialog)
                page.goto(f'http://127.0.0.1:{server.server_port}/?img=test0')
                page.wait_for_function('window.__state?.ann?.instances.length === 2')
                # Actual creation, assignment, save and undo UI.
                page.locator('#newClsName').fill('browser-test')
                page.locator('#btnAddCls').click()
                page.wait_for_function('window.__state.classes.some(c => c.name === "browser-test")')
                page.locator('#instList .inst-row').filter(has_text='#0 ').click()
                page.locator('#instClass').select_option(label='building 建筑')
                page.locator('#btnUndo').click()
                page.locator('#btnSave').click()
                page.wait_for_function('!window.__state.dirty')
                # Pending candidate invokes the real split endpoint; verify remainder survives.
                page.evaluate('''() => { pushUndo(); state.pending={replaceId:0,cands:[{class:"unlabeled",bbox:[10,10,20,40],area:800,score:1,polygons:[[[10,10],[29,10],[29,49],[10,49]]]}]}; showCands(state.pending.cands); }''')
                page.locator('.cand-card').first.click()
                page.wait_for_function('window.__state.ann.instances.length === 3')
                page.evaluate('''() => { setMode("merge"); state.selected=new Set([0,1]); refreshSelInfo(); }''')
                page.get_by_role('button',name='合并所选 (2)').click()
                page.wait_for_function('window.__state.ann.instances.length === 2')
                page.locator('#btnSave').click(); page.wait_for_function('!window.__state.dirty')
                page.evaluate("document.querySelector('#btnTrain').click()")
                page.wait_for_function('document.querySelector("#taskStatus").textContent.includes("查看实验报告")',timeout=60000)
                assert (results/'classifier/evaluation.json').exists()
                page.evaluate('''() => {state.ann.instances.forEach(i => {i.class="unlabeled";delete i.label_source;});markDirty();}''')
                page.locator('#btnClassify').click()
                page.wait_for_function('window.__state.ann.instances.every(i => i.label_source === "model")')
                page.locator('#instList .inst-row').filter(has_text='#0 ').click()
                page.locator('#btnConfirmLabels').click()
                assert page.evaluate('state.ann.instances[0].label_source')=='manual'
                page.locator('#btnSave').click(); page.wait_for_function('!window.__state.dirty')
                page.locator('#btnReport').click()
                page.wait_for_selector('#reportModal',state='visible')
                assert '模型预测待确认 1' in page.locator('#reportBody').inner_text()
                page.locator('#btnCloseReport').click()
                page.locator('#imgList div').filter(has_text='empty.png').click()
                page.wait_for_function('window.__state.imageName === "empty" && window.__state.ann.instances.length === 0')
                assert page.locator('#btnAuto').is_visible()
                # Filtering changes visibility, never removes stored annotations.
                page.locator('#imgList div').filter(has_text='test1.png').click()
                page.wait_for_function('window.__state.ann?.image === "test1.png"')
                page.locator('#minVisibleArea').fill('2500')
                page.locator('#minVisibleArea').dispatch_event('change')
                assert page.locator('#instList .inst-row').count()==1
                assert page.evaluate('state.ann.instances.length')==2
                page.locator('#minVisibleArea').fill('0')
                page.locator('#minVisibleArea').dispatch_event('change')
                page.locator('#instanceFilter').select_option('todo')
                assert page.locator('#instList .inst-row').count()==0
                page.locator('#instanceFilter').select_option('all')
                # Current labeled objects are transferred to a crop; original stays unchanged.
                page.evaluate('''() => setMode("crop")''')
                page.evaluate('''() => cropRegion([0,0,192,128])''')
                page.wait_for_function('window.__state.ann?.image?.startsWith("crop_test1_")')
                assert page.evaluate('state.ann.instances.length')==2
                assert page.locator('#imageGroup').input_value()=='crop'
                # Replacement refinement keeps the class and can expand an old boundary.
                page.evaluate('''() => {pushUndo();state.pending={kind:"refine",replaceId:0,cands:[{class:"unlabeled",bbox:[8,8,54,44],area:2376,score:.99,polygons:[[[8,8],[61,8],[61,51],[8,51]]]}]};showCands(state.pending.cands);}''')
                page.locator('.cand-card').first.click()
                page.wait_for_function('window.__state.ann.instances[0].bbox[0]===8')
                assert page.evaluate('state.ann.instances[0].class')=='airplane 飞机'
                assert page.evaluate('state.ann.instances.length')==2
                page.locator('#btnUndo').click()
                page.locator('#btnSave').click();page.wait_for_function('!state.dirty')
                assert not errors,errors
                assert not dialogs,dialogs
                screenshot=Path(app.BASE)/'results/local_run/browser_smoke.png'
                page.screenshot(path=str(screenshot))
                browser.close()
            print('PASS: Chrome labels, undo, save, split, merge, train, predict, confirm, report, filters, crops, boundary refinement')
        finally:
            server.shutdown()


if __name__=='__main__':main()
