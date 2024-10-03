import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:food_delivery_app/data/services/api_service.dart';
import 'package:food_delivery_app/features/authentication/models/restaurant/basic_info.dart';
import 'package:food_delivery_app/features/authentication/models/restaurant/restaurant.dart';
import 'package:food_delivery_app/features/restaurant/registration/controllers/registration_tab_controller.dart';
import 'package:food_delivery_app/utils/hardcode/hardcode.dart';
import 'package:food_delivery_app/utils/helpers/helper_functions.dart';
import 'package:get/get.dart';

class RegistrationBasicInfoController extends GetxController {
  static RegistrationBasicInfoController get instance => Get.find();

  final formKey = GlobalKey<FormState>();
  final registrationTabController = RegistrationTabController.instance;
  Restaurant? restaurant;
  RestaurantBasicInfo? basicInfo;

  final shopNameController = TextEditingController();
  final streetController = TextEditingController();
  final contactPhoneController = TextEditingController();
  final streetAddressController = TextEditingController();
  final shopType = ''.obs;
  final city = ''.obs;
  final district = ''.obs;

  final RxList<String> hometownOptions = <String>[].obs;
  final RxList<String> cityOptions = <String>[].obs;
  final RxList<String> districtOptions = <String>[].obs;
  final RxList<String> wardOptions = <String>[].obs;

  RegistrationBasicInfoController() {
    restaurant = registrationTabController.restaurant;
    basicInfo = restaurant?.basicInfo;
    cityOptions.value = THardCode.getVietnamLocation().map<String>((city) => city["name"] as String).toList();

    if (basicInfo != null) {
      shopNameController.text = basicInfo?.name ?? '';


      streetAddressController.text = basicInfo?.streetAddress ?? '';
      contactPhoneController.text = basicInfo?.phoneNumber ?? '';
      city.value = basicInfo?.city ?? '';
      district.value = basicInfo?.district ?? '';
    }
    _updateDistrictOptions(city.value);
  }

  void setShopType(String? value) => shopType.value = value ?? '';
  void setCity(String? selectedCity) {
    city.value = selectedCity ?? "";
    district.value = "";
    districtOptions.clear();
    wardOptions.clear();
    _updateDistrictOptions(city.value);
  }

  void _updateDistrictOptions(String cityName) {
    final cityMap = THardCode.getVietnamLocation().firstWhere(
          (city) => city["name"] == cityName,
      orElse: () => {},
    );

    if (cityMap.isNotEmpty) {
      final districts = cityMap["districts"] as List<dynamic>;
      districtOptions.value = districts.map<String>((district) => district["name"] as String).toList();
    }
  }

  void setDistrict(String? selectedDistrict) {
    district.value = selectedDistrict ?? "";
  }

  Future<void> onCallApi() async {
    final basicInfoData = RestaurantBasicInfo(
      name: shopNameController.text,
      phoneNumber: contactPhoneController.text,
      city: city.value,
      district: district.value,
      streetAddress: streetAddressController.text,
    );

    if (basicInfo != null) {
      final [statusCode, headers, data] = await APIService<RestaurantBasicInfo>()
          .update(registrationTabController.restaurant?.id ?? "", basicInfoData, patch: true);
      print([statusCode, headers, data]);
    } else {
      if (restaurant == null) {
        var [statusCode, headers, data] = await APIService<Restaurant>()
            .create({"user": registrationTabController.user?.id});
        print([statusCode, headers, data]);
        if (statusCode == 200 || statusCode == 201) {
          restaurant = data;
          registrationTabController.restaurant = data;
        }
      }
      basicInfoData.restaurant = restaurant?.id;
      final [statusCode, headers, data] = await APIService<RestaurantBasicInfo>()
          .create(basicInfoData);
      print([statusCode, headers, data]);
    }
  }

  void onSave() async {
    if (formKey.currentState?.validate() ?? false) {
      formKey.currentState?.save();
      await onCallApi();
      print('Saving Basic Info');
      Get.snackbar("Success", "Information saved successfully");
    }
  }

  void onContinue() async {
    await onCallApi();
    registrationTabController.setTab();
      formKey.currentState?.save();
    if (formKey.currentState?.validate() ?? false) {
      print('Continuing to next step');
    }
  }

  @override
  void onClose() {
    shopNameController.dispose();
    streetController.dispose();
    contactPhoneController.dispose();
    streetAddressController.dispose();
    super.onClose();
  }
}