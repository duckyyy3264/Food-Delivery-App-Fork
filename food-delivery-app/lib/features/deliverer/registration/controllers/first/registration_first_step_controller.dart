import 'package:flutter/material.dart';
import 'package:food_delivery_app/data/services/deliverer_service.dart';
import 'package:food_delivery_app/features/authentication/models/account/user.dart';
import 'package:food_delivery_app/features/authentication/models/deliverer/deliverer.dart';
import 'package:food_delivery_app/utils/constants/times.dart';
import 'package:food_delivery_app/utils/helpers/helper_functions.dart';
import 'package:get/get.dart';

class RegistrationFirstStepController extends GetxController with GetSingleTickerProviderStateMixin {
  static RegistrationFirstStepController get instance => Get.find();

  int currentTab = 0;
  Rx<bool> isLoading = true.obs;
  User? user;
  Deliverer? deliverer;
  late final TabController tabController;

  @override
  void onInit() {
    super.onInit();
    initialize();
    tabController = TabController(length: 6, vsync: this);
  }

  Future<void> initialize() async {
    isLoading.value = true;
    final [_user, _deliverer] = await DelivererService.getDeliverer(getUser: true);
    user = _user;
    deliverer = _deliverer;
    await Future.delayed(Duration(milliseconds: TTime.init));
    isLoading.value = false;
    $print("HOME TOWN ${deliverer?.basicInfo}");
  }

  void setTab() {
    if(tabController.index + 1 < tabController.length) {
      tabController.animateTo(tabController.index + 1);
    }
  }
}