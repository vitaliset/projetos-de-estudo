
def colision(rocket, meteoro):
    return rocket.x +16 > meteoro.x > rocket.x and rocket.y +16 > meteoro.y > rocket.y

def check_colision(rocket, meteoros):
    for meteoro in meteoros:
        if colision(rocket, meteoro):
            return True
    return False